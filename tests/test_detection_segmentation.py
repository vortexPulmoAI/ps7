"""
Unit tests for detection and segmentation wrappers, patch extractors,
and evaluation metrics (FROC/CPM and Dice/HD95).

NOTE: These tests verify mathematical and spatial pipeline accuracy with known values.
They do not represent benchmark model validation on held-out test data.
"""

import unittest
import numpy as np
from src.detection import (
    NNDetectionDataPrep,
    CandidatePatchExtractor,
    FROCEvaluator
)
from src.segmentation import (
    NNUNetDataPrep,
    NoduleSegmentationInferer,
    SegmentationEvaluator
)


class TestDetectionWrapper(unittest.TestCase):

    def test_coord_to_bbox_conversion(self):
        coord = (100.0, -50.0, 20.0)
        diameter = 10.0
        res = NNDetectionDataPrep.world_coord_to_bbox_3d(coord, diameter, spacing_xyz=(1.0, 1.0, 1.0))

        bbox_mm = res["physical_bbox_mm"]
        self.assertEqual(bbox_mm, [95.0, -55.0, 15.0, 105.0, -45.0, 25.0])
        self.assertEqual(res["diameter_mm"], 10.0)

    def test_patch_extraction(self):
        # 100x100x100 synthetic volume
        vol = np.arange(100*100*100, dtype=np.float32).reshape((100, 100, 100))
        extractor = CandidatePatchExtractor()

        # Extract internal (32x32x32) and contextual (64x64x64) patch at center
        patches = extractor.extract_dual_scale_patches(vol, (50, 50, 50), inner_size=(32, 32, 32), context_size=(64, 64, 64))
        self.assertEqual(patches["inner_patch"].shape, (32, 32, 32))
        self.assertEqual(patches["context_patch"].shape, (64, 64, 64))

        # Test boundary clipping / padding
        edge_patch = extractor.extract_patch_3d(vol, (0, 0, 0), patch_size_zyx=(32, 32, 32), pad_value=-1000.0)
        self.assertEqual(edge_patch.shape, (32, 32, 32))
        self.assertEqual(edge_patch[0, 0, 0], -1000.0)

    def test_cpm_computation(self):
        # If sensitivity at all 7 thresholds is 0.90, CPM must be 0.90
        sens = {0.125: 0.90, 0.25: 0.90, 0.5: 0.90, 1.0: 0.90, 2.0: 0.90, 4.0: 0.90, 8.0: 0.90}
        cpm = FROCEvaluator.compute_cpm(sens)
        self.assertEqual(cpm, 0.90)

        # Monotonically increasing sensitivity curve
        curve = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
        expected_mean = round(float(np.mean(curve)), 4)
        cpm_curve = FROCEvaluator.compute_cpm(curve)
        self.assertEqual(cpm_curve, expected_mean)


class TestSegmentationWrapper(unittest.TestCase):

    def test_polygon_rasterization(self):
        # 10x10 square polygon in 50x50 image
        polygon = [(20, 20), (30, 20), (30, 30), (20, 30)]
        mask = NNUNetDataPrep.polygon_to_mask_2d(polygon, shape_yx=(50, 50))
        self.assertEqual(mask.shape, (50, 50))
        self.assertGreater(np.sum(mask), 50)
        self.assertEqual(mask[25, 25], 1)
        self.assertEqual(mask[5, 5], 0)

    def test_dice_metric(self):
        mask_a = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_b = np.zeros((20, 20, 20), dtype=np.uint8)

        mask_a[5:15, 5:15, 5:15] = 1
        mask_b[5:15, 5:15, 5:15] = 1

        # Identical masks -> Dice = 1.0
        dice_perfect = SegmentationEvaluator.compute_dice(mask_a, mask_b)
        self.assertEqual(dice_perfect, 1.0)

        # Completely disjoint masks -> Dice = 0.0
        mask_c = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_c[0:4, 0:4, 0:4] = 1
        dice_zero = SegmentationEvaluator.compute_dice(mask_a, mask_c)
        self.assertAlmostEqual(dice_zero, 0.0, places=3)

    def test_hd95_metric(self):
        mask_a = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_b = np.zeros((20, 20, 20), dtype=np.uint8)

        mask_a[5:15, 5:15, 5:15] = 1
        mask_b[5:15, 5:15, 5:15] = 1

        # Identical masks -> HD95 = 0.0 mm
        hd95_perfect = SegmentationEvaluator.compute_hd95(mask_a, mask_b)
        self.assertEqual(hd95_perfect, 0.0)

        # Shifted mask by 2 mm
        mask_shifted = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_shifted[5:15, 5:15, 7:17] = 1
        hd95_shifted = SegmentationEvaluator.compute_hd95(mask_a, mask_shifted, voxel_spacing=(1.0, 1.0, 1.0))
        self.assertGreater(hd95_shifted, 0.0)
        self.assertLessEqual(hd95_shifted, 3.0)


if __name__ == "__main__":
    unittest.main()
