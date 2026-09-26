"""
Preprocessing and spatial normalization for 3D Thoracic CT scans.
Handles DICOM patient coordinate transformations, isotropic resampling,
Hounsfield Unit (HU) windowing, and LUNA16 MHD/ZRAW lung mask loading.
"""

import os
import zlib
import numpy as np
import scipy.ndimage


class AffineTransform:
    """
    Manages the 4x4 affine coordinate transformation matrix mapping discrete
    voxel array indices (i, j, k) to physical Patient Coordinates (X, Y, Z) in mm.

    T_world = [
        [R11 * sx, R12 * sy, R13 * sz, Tx],
        [R21 * sx, R22 * sy, R23 * sz, Ty],
        [R31 * sx, R32 * sy, R33 * sz, Tz],
        [0,        0,        0,        1 ]
    ]
    """

    def __init__(
        self,
        image_position_patient=(0.0, 0.0, 0.0),
        image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(1.0, 1.0),
        slice_thickness=1.0,
    ):
        """
        Args:
            image_position_patient: (Tx, Ty, Tz) physical coordinates of the upper-left voxel
            image_orientation_patient: (rx1, rx2, rx3, ry1, ry2, ry3) row and column direction cosines
            pixel_spacing: (sx, sy) in-plane voxel dimensions in mm (row_spacing, col_spacing)
            slice_thickness: sz longitudinal slice spacing in mm
        """
        self.image_position = np.array(image_position_patient, dtype=np.float64)
        iop = np.array(image_orientation_patient, dtype=np.float64)

        row_cosine = iop[0:3]
        col_cosine = iop[3:6]
        # Cross product yields the normal slice direction vector
        slice_cosine = np.cross(row_cosine, col_cosine)
        slice_norm = np.linalg.norm(slice_cosine)
        if slice_norm > 1e-6:
            slice_cosine = slice_cosine / slice_norm

        sx, sy = float(pixel_spacing[0]), float(pixel_spacing[1])
        sz = float(slice_thickness)
        self.spacing = np.array([sx, sy, sz], dtype=np.float64)

        # Assemble 3x3 rotation/scaling matrix
        R = np.zeros((3, 3), dtype=np.float64)
        R[:, 0] = row_cosine * sx
        R[:, 1] = col_cosine * sy
        R[:, 2] = slice_cosine * sz

        # Assemble full 4x4 affine matrix
        self.affine = np.eye(4, dtype=np.float64)
        self.affine[0:3, 0:3] = R
        self.affine[0:3, 3] = self.image_position
        self.inv_affine = np.linalg.inv(self.affine)

    def voxel_to_world(self, voxel_coords):
        """
        Transforms discrete array index (i, j, k) to physical Patient Coordinate (X, Y, Z) in mm.
        Accepts single coordinate tuple or (N, 3) array.
        """
        coords = np.atleast_2d(voxel_coords)
        homogeneous = np.hstack([coords, np.ones((len(coords), 1), dtype=np.float64)])
        world = homogeneous @ self.affine.T
        res = world[:, 0:3]
        return res[0] if np.asarray(voxel_coords).ndim == 1 else res

    def world_to_voxel(self, world_coords):
        """
        Transforms physical Patient Coordinate (X, Y, Z) in mm to array index (i, j, k).
        Accepts single coordinate tuple or (N, 3) array.
        """
        coords = np.atleast_2d(world_coords)
        homogeneous = np.hstack([coords, np.ones((len(coords), 1), dtype=np.float64)])
        voxel = homogeneous @ self.inv_affine.T
        res = voxel[:, 0:3]
        return res[0] if np.asarray(world_coords).ndim == 1 else res


def resample_isotropic(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0), is_mask=False):
    """
    Resamples a 3D CT volume or segmentation mask to isotropic spacing.

    Args:
        volume: 3D numpy array [dim_z, dim_y, dim_x]
        current_spacing: (sz, sy, sx) physical spacing in mm
        target_spacing: (tz, ty, tx) target physical spacing in mm (default 1.0 mm^3)
        is_mask: If True, uses nearest-neighbor interpolation (order=0) to prevent
                 label contamination; otherwise uses 3rd-order spline (order=3).

    Returns:
        resampled_volume: 3D numpy array resampled to target spacing
        new_spacing: tuple of target spacing
    """
    current_spacing = np.array(current_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)
    zoom_factors = current_spacing / target_spacing

    order = 0 if is_mask else 3
    resampled = scipy.ndimage.zoom(volume, zoom_factors, order=order, mode="nearest")
    if is_mask:
        resampled = (resampled > 0.5).astype(volume.dtype)

    return resampled, tuple(target_spacing)


def apply_hu_window(volume, window="lung", normalize=True):
    """
    Applies Hounsfield Unit (HU) window clipping and intensity normalization.

    Windows:
        - "lung": Center = -600 HU, Width = 1500 HU -> [-1350, +150 HU]
        - "lung_screening": Center = -300 HU, Width = 1400 HU -> [-1000, +400 HU]
        - "soft_tissue" / "mediastinum": Center = +40 HU, Width = 400 HU -> [-160, +240 HU]
        - "bone": Center = +400 HU, Width = 1800 HU -> [-500, +1300 HU]

    Args:
        volume: numpy array in Hounsfield Units
        window: window preset name or tuple (vmin, vmax)
        normalize: If True, scales clipped values to [0.0, 1.0]

    Returns:
        windowed: clipped and optionally normalized numpy array
    """
    presets = {
        "lung": (-1350.0, 150.0),
        "lung_screening": (-1000.0, 400.0),
        "soft_tissue": (-160.0, 240.0),
        "mediastinum": (-160.0, 240.0),
        "bone": (-500.0, 1300.0),
    }

    if isinstance(window, str):
        vmin, vmax = presets.get(window.lower(), (-1000.0, 400.0))
    else:
        vmin, vmax = float(window[0]), float(window[1])

    clipped = np.clip(volume, vmin, vmax)
    if normalize:
        return (clipped - vmin) / (vmax - vmin)
    return clipped


def load_mhd_mask(mhd_path):
    """
    Parses a MetaImage (.mhd) header and loads the associated (.zraw or .raw)
    binary mask (e.g. from dataset/seg-lungs-LUNA16).

    Args:
        mhd_path: path to the .mhd file

    Returns:
        mask_array: 3D numpy array [dim_z, dim_y, dim_x]
        metadata: dictionary of header properties (spacing, origin, transform, etc.)
    """
    if not os.path.exists(mhd_path):
        raise FileNotFoundError(f"MHD file not found: {mhd_path}")

    metadata = {}
    with open(mhd_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = [p.strip() for p in line.split("=", 1)]
            metadata[key] = val

    # Extract dimensions and spacing
    dims = [int(x) for x in metadata.get("DimSize", "512 512 1").split()]
    spacing = [float(x) for x in metadata.get("ElementSpacing", "1 1 1").split()]
    offset = [float(x) for x in metadata.get("Offset", "0 0 0").split()]

    metadata["dim_size_xyz"] = dims
    metadata["spacing_xyz"] = spacing
    metadata["offset_xyz"] = offset

    raw_filename = metadata.get("ElementDataFile", "")
    raw_path = os.path.join(os.path.dirname(mhd_path), raw_filename)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Associated raw data file not found: {raw_path}")

    # Determine dtype
    elem_type = metadata.get("ElementType", "MET_SHORT")
    dtype_map = {
        "MET_CHAR": np.int8,
        "MET_UCHAR": np.uint8,
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_INT": np.int32,
        "MET_UINT": np.uint32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    dtype = dtype_map.get(elem_type, np.int16)

    # Read binary data
    with open(raw_path, "rb") as f:
        raw_bytes = f.read()

    is_compressed = metadata.get("CompressedData", "False").lower() == "true"
    if is_compressed:
        decompressed_bytes = zlib.decompress(raw_bytes)
        arr = np.frombuffer(decompressed_bytes, dtype=dtype)
    else:
        arr = np.frombuffer(raw_bytes, dtype=dtype)

    # Shape in MHD is DimSize: [dimX, dimY, dimZ], so in numpy array indexing: [dimZ, dimY, dimX]
    dim_x, dim_y, dim_z = dims[0], dims[1], dims[2]
    expected_voxels = dim_x * dim_y * dim_z
    if arr.size != expected_voxels:
        raise ValueError(
            f"Voxel count mismatch: expected {expected_voxels} ({dims}), got {arr.size}"
        )

    mask_3d = arr.reshape((dim_z, dim_y, dim_x))
    return mask_3d, metadata
