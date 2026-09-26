"""
Automated LUNA16 Volumetric Scan Ingestion and Patch Extraction Engine.
Ingests LUNA16 CT subsets (e.g. subset0/, subset1/, etc.), pairs each scan with its
corresponding seg-lungs-LUNA16 mask, cross-references with annotations.csv and candidates.csv,
and generates isotropic 3D candidate subvolumes (32^3 and 64^3 mm) for nnDetection and FPR training.
"""

import os
import glob
import numpy as np
from .preprocessing import load_mhd_mask, resample_isotropic, apply_hu_window, AffineTransform
from .dataset_parser import LUNA16AnnotationParser, LUNA16CandidateParser, LUNALungMaskLoader
from .detection import CandidatePatchExtractor, NNDetectionDataPrep


class LUNA16SubsetIngestor:
    """
    Manages automated ingestion and preparation of LUNA16 CT scan subsets.
    """

    def __init__(self, dataset_root="dataset"):
        self.dataset_root = dataset_root
        self.annotations_path = os.path.join(dataset_root, "annotations.csv")
        self.candidates_path = os.path.join(dataset_root, "candidates.csv")
        self.lungs_dir = os.path.join(dataset_root, "seg-lungs-LUNA16")

        self.anno_parser = (
            LUNA16AnnotationParser(self.annotations_path)
            if os.path.exists(self.annotations_path)
            else None
        )
        self.cand_parser = (
            LUNA16CandidateParser(self.candidates_path)
            if os.path.exists(self.candidates_path)
            else None
        )
        self.mask_loader = (
            LUNALungMaskLoader(self.lungs_dir)
            if os.path.exists(self.lungs_dir)
            else None
        )
        self.patch_extractor = CandidatePatchExtractor()

    def discover_subset_scans(self, subset_name="subset0"):
        """
        Discovers all available .mhd CT scans within a subset folder.
        """
        subset_path = os.path.join(self.dataset_root, subset_name)
        if not os.path.exists(subset_path):
            matches = glob.glob(os.path.join(self.dataset_root, "**", subset_name, "*.mhd"), recursive=True)
            return sorted(matches) if matches else []

        direct = glob.glob(os.path.join(subset_path, "*.mhd"))
        if direct:
            return sorted(direct)
        return sorted(glob.glob(os.path.join(subset_path, "**", "*.mhd"), recursive=True))

    def load_scan(self, mhd_path):
        """
        Loads CT volume, spatial metadata, and associated lung mask.

        Returns:
            dict containing:
                series_uid: str
                volume_hu: 3D numpy array [dimZ, dimY, dimX]
                spacing_zyx: (sz, sy, sx)
                origin_xyz: (ox, oy, oz)
                lung_mask: 3D numpy array or None
                affine: AffineTransform instance
        """
        # Try SimpleITK first if available for fastest reading, fallback to load_mhd_mask
        series_uid = os.path.splitext(os.path.basename(mhd_path))[0]
        lung_mask = None

        try:
            import SimpleITK as sitk
            itk_img = sitk.ReadImage(mhd_path)
            vol_hu = sitk.GetArrayFromImage(itk_img)  # [Z, Y, X]
            spacing_xyz = itk_img.GetSpacing()
            origin_xyz = itk_img.GetOrigin()
            direction = itk_img.GetDirection()
            spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
            affine = AffineTransform(
                image_position_patient=origin_xyz,
                image_orientation_patient=direction[:6],
                pixel_spacing=(spacing_xyz[0], spacing_xyz[1]),
                slice_thickness=spacing_xyz[2],
            )
        except Exception:
            vol_hu, meta = load_mhd_mask(mhd_path)
            spacing_xyz = meta["spacing_xyz"]
            origin_xyz = meta["offset_xyz"]
            spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
            affine = AffineTransform(
                image_position_patient=origin_xyz,
                pixel_spacing=(spacing_xyz[0], spacing_xyz[1]),
                slice_thickness=spacing_xyz[2],
            )

        # Load paired lung mask if available
        if self.mask_loader:
            try:
                lung_mask, _ = self.mask_loader.load_mask_for_series(series_uid)
            except Exception:
                lung_mask = None

        return {
            "series_uid": series_uid,
            "volume_hu": vol_hu,
            "spacing_zyx": spacing_zyx,
            "origin_xyz": origin_xyz,
            "lung_mask": lung_mask,
            "affine": affine,
        }

    def extract_nodule_patches_for_series(
        self,
        mhd_path,
        target_spacing=(1.0, 1.0, 1.0),
        inner_size=(32, 32, 32),
        context_size=(64, 64, 64)
    ):
        """
        Loads scan, resamples to isotropic spacing, applies lung windowing,
        and extracts 3D patches for all true nodules annotated in annotations.csv.

        Returns:
            list of dicts containing patch arrays, metadata, and nodule labels
        """
        scan = self.load_scan(mhd_path)
        series_uid = scan["series_uid"]
        vol_hu = scan["volume_hu"]
        spacing_zyx = scan["spacing_zyx"]
        affine = scan["affine"]

        # Resample CT to isotropic
        resampled_vol, _ = resample_isotropic(vol_hu, spacing_zyx, target_spacing=target_spacing)
        # Apply standard lung window [-1000, +400 HU]
        normalized_vol = apply_hu_window(resampled_vol, window="lung_screening", normalize=True)

        nodules = self.anno_parser.get_by_series(series_uid) if self.anno_parser else []
        extracted_samples = []

        for nod in nodules:
            world_coord = (nod["coord_x"], nod["coord_y"], nod["coord_z"])
            # Map world coordinate (X, Y, Z) to resampled voxel index
            # Physical offset from origin:
            cx, cy, cz = world_coord
            ox, oy, oz = scan["origin_xyz"]

            # Voxel index in original volume
            sx, sy, sz = spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]
            vox_x = (cx - ox) / sx
            vox_y = (cy - oy) / sy
            vox_z = (cz - oz) / sz

            # Scale to resampled isotropic grid
            iso_z = vox_z * (sz / target_spacing[0])
            iso_y = vox_y * (sy / target_spacing[1])
            iso_x = vox_x * (sx / target_spacing[2])

            center_zyx = (iso_z, iso_y, iso_x)

            patches = self.patch_extractor.extract_dual_scale_patches(
                normalized_vol,
                center_zyx,
                inner_size=inner_size,
                context_size=context_size,
                pad_value=0.0
            )

            extracted_samples.append({
                "series_uid": series_uid,
                "world_coord": world_coord,
                "diameter_mm": nod["diameter_mm"],
                "center_zyx": center_zyx,
                "inner_patch": patches["inner_patch"],
                "context_patch": patches["context_patch"],
                "label": 1,  # True positive nodule
            })

        return extracted_samples
