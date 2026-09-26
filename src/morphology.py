"""
Quantitative 3D morphological profiling for pulmonary nodules.
Calculates true physical volume, principal axes via the 3D second-moment inertia tensor,
maximal transverse slice long/short-axis diameters, sphericity index, and spiculation score.
"""

import numpy as np


def calculate_volume(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Computes true physical volume in mm^3 by integrating active mask voxels
    scaled by the unit voxel volume:
        V = (sum M(x, y, z)) * (sx * sy * sz)

    Args:
        mask: 2D or 3D binary numpy array [dim_z, dim_y, dim_x] or [dim_y, dim_x]
        voxel_spacing: (sz, sy, sx) or (sy, sx) in mm

    Returns:
        float: physical volume in mm^3 (or area in mm^2 if 2D)
    """
    active_voxels = int(np.count_nonzero(mask))
    unit_volume = float(np.prod(voxel_spacing))
    return active_voxels * unit_volume


def calculate_diameters_and_axes(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculates 3D principal axes via the second-moment inertia tensor,
    and extracts transverse plane (axial slice with maximal area)
    long-axis, short-axis, and mean diameter according to Fleischner/Lung-RADS standards.

    Args:
        mask: 3D binary array [dim_z, dim_y, dim_x]
        voxel_spacing: (sz, sy, sx) in mm

    Returns:
        dict containing:
            d_long_mm: longest transverse diameter
            d_short_mm: perpendicular short-axis diameter
            d_mean_mm: mean transverse diameter (d_long + d_short) / 2
            principal_axes_mm: (axis1, axis2, axis3) 3D equivalent ellipsoid full diameters
            centroid_mm: (cz, cy, cx) center of mass in physical coordinates
            max_slice_index: z-index of maximal cross-sectional slice
    """
    active_indices = np.argwhere(mask > 0)
    if len(active_indices) < 4:
        return {
            "d_long_mm": 0.0,
            "d_short_mm": 0.0,
            "d_mean_mm": 0.0,
            "principal_axes_mm": (0.0, 0.0, 0.0),
            "centroid_mm": (0.0, 0.0, 0.0),
            "max_slice_index": 0,
        }

    spacing = np.array(voxel_spacing, dtype=np.float64)
    # Convert voxel coordinates to physical mm
    coords_mm = active_indices.astype(np.float64) * spacing
    centroid_mm = np.mean(coords_mm, axis=0)

    # 3D second-moment inertia / covariance tensor: I_ij = (1/N) * sum (r_i - c_i)(r_j - c_j)
    centered = coords_mm - centroid_mm
    cov_3d = (centered.T @ centered) / len(centered)
    eigenvalues_3d = np.linalg.eigvalsh(cov_3d)
    eigenvalues_3d = np.maximum(eigenvalues_3d, 0.0)
    # For a uniform ellipsoid, eigenvalue lambda = (radius^2) / 5, so full axis = 2 * sqrt(5 * lambda)
    # Alternatively for bounding standard deviation: 4 * sqrt(lambda) (2 standard deviations each side)
    principal_axes_mm = tuple(np.sort(2.0 * np.sqrt(5.0 * eigenvalues_3d))[::-1])

    # Find transverse slice (z-axis) with maximum cross-sectional area
    z_coords = active_indices[:, 0]
    unique_z, counts_z = np.unique(z_coords, return_counts=True)
    max_z = unique_z[np.argmax(counts_z)]

    # Extract 2D points on maximal transverse slice in physical (y, x) mm
    slice_mask = (z_coords == max_z)
    slice_points_mm = coords_mm[slice_mask, 1:3]  # [y, x] in mm

    if len(slice_points_mm) >= 3:
        slice_center = np.mean(slice_points_mm, axis=0)
        centered_2d = slice_points_mm - slice_center
        cov_2d = (centered_2d.T @ centered_2d) / len(centered_2d)
        evals_2d = np.linalg.eigvalsh(cov_2d)
        evals_2d = np.maximum(evals_2d, 0.0)
        # Uniform ellipse axes = 2 * sqrt(4 * lambda)
        d_long = 4.0 * np.sqrt(evals_2d[1])
        d_short = 4.0 * np.sqrt(evals_2d[0])
    else:
        # Fallback to direct voxel span
        y_span = (np.max(slice_points_mm[:, 0]) - np.min(slice_points_mm[:, 0])) + spacing[1]
        x_span = (np.max(slice_points_mm[:, 1]) - np.min(slice_points_mm[:, 1])) + spacing[2]
        d_long = max(y_span, x_span)
        d_short = min(y_span, x_span)

    d_mean = (d_long + d_short) / 2.0

    return {
        "d_long_mm": round(float(d_long), 1),
        "d_short_mm": round(float(d_short), 1),
        "d_mean_mm": round(float(d_mean), 1),
        "principal_axes_mm": tuple(round(float(a), 2) for a in principal_axes_mm),
        "centroid_mm": tuple(round(float(c), 2) for c in centroid_mm),
        "max_slice_index": int(max_z),
    }


def calculate_sphericity(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Computes 3D sphericity index Psi:
        Psi = (pi^(1/3) * (6 * V)^(2/3)) / A_surface

    Where V is physical volume in mm^3, and A_surface is outer surface area in mm^2.
    For an ideal sphere, Psi = 1.0. Irregular, invasive lesions exhibit Psi << 1.0.

    Args:
        mask: 3D binary array [dim_z, dim_y, dim_x]
        voxel_spacing: (sz, sy, sx) in mm

    Returns:
        float: sphericity index in range (0.0, 1.0]
    """
    volume = calculate_volume(mask, voxel_spacing)
    if volume <= 0.0:
        return 0.0

    # Use marching cubes to extract the isosurface mesh and compute accurate surface area
    try:
        import skimage.measure
        verts, faces, _, _ = skimage.measure.marching_cubes(mask.astype(float), level=0.5, spacing=voxel_spacing)
        surface_area = float(skimage.measure.mesh_surface_area(verts, faces))
    except Exception:
        # Fallback: Count exposed voxel face transitions scaled by unit face areas
        sz, sy, sx = voxel_spacing
        face_xy = sx * sy
        face_xz = sx * sz
        face_yz = sy * sz

        m = mask.astype(bool)
        pad = np.pad(m, 1, mode="constant", constant_values=False)
        diff_z = np.logical_xor(pad[:-1, 1:-1, 1:-1], pad[1:, 1:-1, 1:-1])
        diff_y = np.logical_xor(pad[1:-1, :-1, 1:-1], pad[1:-1, 1:, 1:-1])
        diff_x = np.logical_xor(pad[1:-1, 1:-1, :-1], pad[1:-1, 1:-1, 1:])

        # Note: Manhattan voxel surface overestimates smooth sphere surface area by ~1.5x (Crofton's formula)
        surface_area = (
            np.sum(diff_z) * face_xy
            + np.sum(diff_y) * face_xz
            + np.sum(diff_x) * face_yz
        ) / 1.5

    if surface_area <= 0.0:
        return 0.0

    psi = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface_area
    return float(np.clip(psi, 0.0, 1.0))


def calculate_spiculation_index(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Quantifies surface spiculation and lobulation irregularity using the normalized
    radial distance variance from the nodule centroid:
        sigma_R^2 / mu_R^2

    Smooth, round benign nodules present low variance (~0.01 - 0.04),
    whereas stellate, spiculated malignant lesions exhibit high variance (> 0.08).

    Args:
        mask: 3D binary array
        voxel_spacing: (sz, sy, sx) in mm

    Returns:
        float: normalized spiculation variance index
    """
    active_indices = np.argwhere(mask > 0)
    if len(active_indices) < 10:
        return 0.0

    spacing = np.array(voxel_spacing, dtype=np.float64)
    coords_mm = active_indices.astype(np.float64) * spacing
    centroid_mm = np.mean(coords_mm, axis=0)

    # Find boundary voxels (voxels with at least one 0 in 6-connected neighborhood)
    m = mask.astype(bool)
    pad = np.pad(m, 1, mode="constant", constant_values=False)
    eroded = (
        pad[:-2, 1:-1, 1:-1]
        & pad[2:, 1:-1, 1:-1]
        & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 2:, 1:-1]
        & pad[1:-1, 1:-1, :-2]
        & pad[1:-1, 1:-1, 2:]
    )
    boundary = m & (~eroded)
    boundary_indices = np.argwhere(boundary)

    if len(boundary_indices) < 6:
        boundary_coords = coords_mm
    else:
        boundary_coords = boundary_indices.astype(np.float64) * spacing

    # Compute radial distance distribution from centroid
    radial_distances = np.linalg.norm(boundary_coords - centroid_mm, axis=1)
    mean_r = np.mean(radial_distances)
    if mean_r <= 1e-4:
        return 0.0

    var_r = np.var(radial_distances)
    spiculation_index = var_r / (mean_r ** 2)
    return round(float(spiculation_index), 4)
