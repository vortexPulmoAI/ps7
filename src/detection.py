"""
3D Candidate Detection and False Positive Reduction (FPR) wrapper module.
Provides dataset preparation for nnDetection (3D Retina U-Net), dual-scale candidate patch
extraction (32^3 mm and 64^3 mm crops), and LUNA16 Free-Response ROC (FROC / CPM) evaluation.

NOTE: This module provides the data preparation, coordinate mapping, patch extraction,
and metric evaluation pipeline. Training and model checkpoint optimization are executed
separately using dedicated GPU hardware.
"""

import json
import numpy as np


class NNDetectionDataPrep:
    """
    Translates physical world coordinates from LUNA16 annotations and candidate proposals
    into 3D bounding box labels formatted for the nnDetection framework.
    """

    @staticmethod
    def world_coord_to_bbox_3d(coord_xyz, diameter_mm, affine_transform=None, spacing_xyz=(1.0, 1.0, 1.0)):
        """
        Converts centroid (X, Y, Z) and diameter_mm into 3D bounding box coordinates:
        [x_min, y_min, z_min, x_max, y_max, z_max] in physical mm or voxel units.

        Args:
            coord_xyz: (cx, cy, cz) in mm
            diameter_mm: nodule diameter in mm
            affine_transform: optional AffineTransform instance to map to voxel indices
            spacing_xyz: (sx, sy, sz) voxel spacing if mapping directly

        Returns:
            dict containing physical_bbox_mm and voxel_bbox
        """
        cx, cy, cz = coord_xyz
        r = diameter_mm / 2.0
        bbox_mm = [cx - r, cy - r, cz - r, cx + r, cy + r, cz + r]

        if affine_transform is not None:
            v_min = affine_transform.world_to_voxel([bbox_mm[0], bbox_mm[1], bbox_mm[2]])
            v_max = affine_transform.world_to_voxel([bbox_mm[3], bbox_mm[4], bbox_mm[5]])
            voxel_bbox = [
                min(v_min[0], v_max[0]),
                min(v_min[1], v_max[1]),
                min(v_min[2], v_max[2]),
                max(v_min[0], v_max[0]),
                max(v_min[1], v_max[1]),
                max(v_min[2], v_max[2]),
            ]
        else:
            # Approximate voxel bbox using spacing
            sx, sy, sz = spacing_xyz
            voxel_bbox = [
                (cx - r) / sx,
                (cy - r) / sy,
                (cz - r) / sz,
                (cx + r) / sx,
                (cy + r) / sy,
                (cz + r) / sz,
            ]

        return {
            "physical_bbox_mm": [round(float(v), 2) for v in bbox_mm],
            "voxel_bbox": [round(float(v), 2) for v in voxel_bbox],
            "diameter_mm": float(diameter_mm),
        }

    @staticmethod
    def generate_task_json(task_name, num_cases, modality="CT", labels=None):
        """
        Generates dataset.json specification for nnDetection task configurations.
        """
        if labels is None:
            labels = {"0": "nodule"}
        return {
            "name": task_name,
            "description": "LUNA16 3D Pulmonary Nodule Detection Benchmark",
            "modality": {"0": modality},
            "labels": labels,
            "numTraining": int(num_cases),
            "file_ending": ".nii.gz",
        }


class CandidatePatchExtractor:
    """
    Extracts dual-scale 3D sub-volume patches centered on candidate centroids
    for False Positive Reduction (FPR) classification:
        - Internal patch (e.g. 32x32x32 mm): margins and internal attenuation
        - Contextual patch (e.g. 64x64x64 mm): relationship to vessels, pleura, and chest wall
    """

    @staticmethod
    def extract_patch_3d(volume_3d, center_zyx, patch_size_zyx=(32, 32, 32), pad_value=0.0):
        """
        Extracts a 3D subvolume around center_zyx [z, y, x] with specified shape.
        Pads with pad_value if patch exceeds array boundaries.

        Args:
            volume_3d: 3D numpy array [dim_z, dim_y, dim_x]
            center_zyx: (cz, cy, cx) center voxel coordinates (integer or float)
            patch_size_zyx: (sz, sy, sx) dimensions of the target patch
            pad_value: constant value for out-of-boundary padding

        Returns:
            patch: 3D numpy array of shape patch_size_zyx
        """
        cz, cy, cx = [int(round(c)) for c in center_zyx]
        pz, py, px = patch_size_zyx

        z_min, z_max = cz - pz // 2, cz + (pz - pz // 2)
        y_min, y_max = cy - py // 2, cy + (py - py // 2)
        x_min, x_max = cx - px // 2, cx + (px - px // 2)

        # Output patch initialized with pad value
        patch = np.full(patch_size_zyx, pad_value, dtype=volume_3d.dtype)

        # Calculate overlap in input volume
        src_z_min, src_z_max = max(0, z_min), min(volume_3d.shape[0], z_max)
        src_y_min, src_y_max = max(0, y_min), min(volume_3d.shape[1], y_max)
        src_x_min, src_x_max = max(0, x_min), min(volume_3d.shape[2], x_max)

        if src_z_min >= src_z_max or src_y_min >= src_y_max or src_x_min >= src_x_max:
            return patch

        # Corresponding slice in target patch
        dst_z_min, dst_z_max = src_z_min - z_min, src_z_max - z_min
        dst_y_min, dst_y_max = src_y_min - y_min, src_y_max - y_min
        dst_x_min, dst_x_max = src_x_min - x_min, src_x_max - x_min

        patch[dst_z_min:dst_z_max, dst_y_min:dst_y_max, dst_x_min:dst_x_max] = (
            volume_3d[src_z_min:src_z_max, src_y_min:src_y_max, src_x_min:src_x_max]
        )

        return patch

    def extract_dual_scale_patches(
        self,
        volume_3d,
        center_zyx,
        inner_size=(32, 32, 32),
        context_size=(64, 64, 64),
        pad_value=-1000.0,
    ):
        """
        Extracts both internal and contextual 3D patches for a candidate proposal.
        """
        inner_patch = self.extract_patch_3d(volume_3d, center_zyx, inner_size, pad_value=pad_value)
        context_patch = self.extract_patch_3d(volume_3d, center_zyx, context_size, pad_value=pad_value)
        return {
            "inner_patch": inner_patch,
            "context_patch": context_patch,
        }


class FROCEvaluator:
    """
    Evaluates Free-Response Receiver Operating Characteristic (FROC) curve
    and computes the Competitive Performance Metric (CPM) for nodule detection.
    CPM = average sensitivity across 7 predefined FP/scan rates:
          k in {1/8, 1/4, 1/2, 1, 2, 4, 8} (i.e. 0.125, 0.25, 0.5, 1, 2, 4, 8)
    """

    FROCTHRESHOLDS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    @classmethod
    def compute_cpm(cls, sensitivities_at_thresholds):
        """
        Computes CPM as the arithmetic mean of sensitivities at the 7 standard thresholds:
            CPM = (1/7) * sum(Sensitivity_at_k)

        Args:
            sensitivities_at_thresholds: dict mapping threshold -> sensitivity (float in [0, 1])
                                         or list/tuple of 7 sensitivities in order.

        Returns:
            float: CPM score in range [0.0, 1.0]
        """
        if isinstance(sensitivities_at_thresholds, dict):
            values = [float(sensitivities_at_thresholds.get(k, 0.0)) for k in cls.FROCTHRESHOLDS]
        else:
            values = [float(v) for v in sensitivities_at_thresholds]
            if len(values) != 7:
                raise ValueError(f"Expected 7 sensitivity values for thresholds {cls.FROCTHRESHOLDS}")

        return round(float(np.mean(values)), 4)
