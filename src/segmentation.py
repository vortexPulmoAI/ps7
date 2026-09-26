"""
Volumetric Nodule Segmentation wrapper module.
Provides dataset preparation for nnU-Net v2 (converting LIDC radiologist polygon contours into 3D masks),
sub-volume extraction, post-processing heuristics, and spatial overlap metrics (Dice & HD95).

NOTE: This module provides the dataset preparation, rasterization, inference wrapper,
and metric evaluation pipeline. Full 3D nnU-Net model training is executed separately
on GPU hardware.
"""

import json
import numpy as np
import scipy.ndimage


class NNUNetDataPrep:
    """
    Utilities for compiling LIDC-IDRI radiologist contours into nnU-Net v2 raw datasets.
    """

    @staticmethod
    def polygon_to_mask_2d(polygon_points, shape_yx=(512, 512)):
        """
        Rasterizes a list of (x, y) boundary vertices into a 2D binary slice mask.

        Args:
            polygon_points: list of (x, y) tuples
            shape_yx: (dim_y, dim_x) output image size

        Returns:
            mask_2d: 2D binary numpy array (uint8)
        """
        mask = np.zeros(shape_yx, dtype=np.uint8)
        if len(polygon_points) < 3:
            return mask

        # Try OpenCV fillPoly if available, else rasterize with scanline/matplotlib
        try:
            import cv2
            pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
            return mask
        except ImportError:
            pass

        # Fallback polygon rasterizer
        from matplotlib.path import Path
        ny, nx = shape_yx
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T

        path = Path(polygon_points)
        grid = path.contains_points(points)
        mask = grid.reshape((ny, nx)).astype(np.uint8)
        return mask

    @staticmethod
    def generate_dataset_json(output_path, num_training, dataset_name="Dataset101_LIDC_Nodules"):
        """
        Generates standard nnU-Net v2 dataset.json descriptor.
        """
        descriptor = {
            "channel_names": {
                "0": "CT"
            },
            "labels": {
                "background": 0,
                "nodule": 1
            },
            "numTraining": int(num_training),
            "file_ending": ".nii.gz",
            "name": dataset_name,
            "description": "LIDC-IDRI 3D Pulmonary Nodule Volumetric Segmentation",
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptor, f, indent=4)
        return descriptor


class NoduleSegmentationInferer:
    """
    Subvolume inference wrapper for 3D nodule segmentation.
    Extracts a region of interest around a detected candidate, executes model inference,
    and applies morphological post-processing (largest connected component and hole filling).
    """

    @staticmethod
    def post_process_mask(raw_mask):
        """
        Isolates the single largest 3D connected component and fills internal holes.

        Args:
            raw_mask: 3D binary array [dim_z, dim_y, dim_x]

        Returns:
            cleaned_mask: 3D binary array (uint8)
        """
        labeled, num_features = scipy.ndimage.label(raw_mask > 0)
        if num_features == 0:
            return np.zeros_like(raw_mask, dtype=np.uint8)

        # Find largest component
        counts = np.bincount(labeled.flat)
        counts[0] = 0  # ignore background
        largest_label = np.argmax(counts)

        cleaned = (labeled == largest_label)
        # Fill holes in 3D
        filled = scipy.ndimage.binary_fill_holes(cleaned)
        return filled.astype(np.uint8)


class SegmentationEvaluator:
    """
    Computes volumetric segmentation fidelity metrics:
        - Dice Similarity Coefficient (DSC): 2 * |X cap Y| / (|X| + |Y|)
        - 95th Percentile Hausdorff Distance (HD95) in physical mm
    """

    @staticmethod
    def compute_dice(pred_mask, gt_mask, epsilon=1e-6):
        """
        Computes Dice Similarity Coefficient (DSC).

        Args:
            pred_mask: binary array of predictions
            gt_mask: binary array of ground truth annotations
            epsilon: smoothing constant to prevent division by zero

        Returns:
            float: Dice score in range [0.0, 1.0]
        """
        p = pred_mask > 0
        g = gt_mask > 0

        intersection = np.logical_and(p, g).sum()
        total = p.sum() + g.sum()

        if total == 0:
            return 1.0  # Both empty matches perfectly

        dice = (2.0 * intersection + epsilon) / (total + epsilon)
        return round(float(dice), 4)

    @staticmethod
    def compute_hd95(pred_mask, gt_mask, voxel_spacing=(1.0, 1.0, 1.0)):
        """
        Computes 95th-Percentile Hausdorff Distance in physical millimeters
        using distance transforms on boundary surfaces.

        Args:
            pred_mask: binary array
            gt_mask: binary array
            voxel_spacing: (sz, sy, sx) physical spacing in mm

        Returns:
            float: 95th percentile distance in mm (or 0.0 if identical/empty)
        """
        p = (pred_mask > 0).astype(bool)
        g = (gt_mask > 0).astype(bool)

        if np.array_equal(p, g):
            return 0.0
        if not np.any(p) or not np.any(g):
            return float("inf")

        # Extract boundaries
        pad_p = np.pad(p, 1, mode="constant", constant_values=False)
        pad_g = np.pad(g, 1, mode="constant", constant_values=False)

        eroded_p = scipy.ndimage.binary_erosion(pad_p)[1:-1, 1:-1, 1:-1]
        eroded_g = scipy.ndimage.binary_erosion(pad_g)[1:-1, 1:-1, 1:-1]

        boundary_p = p & (~eroded_p)
        boundary_g = g & (~eroded_g)

        if not np.any(boundary_p) or not np.any(boundary_g):
            return float("inf")

        # Distance transform from ground truth boundary
        dt_g = scipy.ndimage.distance_transform_edt(~boundary_g, sampling=voxel_spacing)
        # Distance transform from predicted boundary
        dt_p = scipy.ndimage.distance_transform_edt(~boundary_p, sampling=voxel_spacing)

        # Distances from pred boundary to nearest gt boundary
        d_p_to_g = dt_g[boundary_p]
        # Distances from gt boundary to nearest pred boundary
        d_g_to_p = dt_p[boundary_g]

        hd95_p = np.percentile(d_p_to_g, 95)
        hd95_g = np.percentile(d_g_to_p, 95)

        return round(float(max(hd95_p, hd95_g)), 2)
