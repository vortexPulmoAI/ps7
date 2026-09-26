"""
Unit tests for morphology calculations and Hounsfield density decomposition.
"""

import unittest
import numpy as np
from src.morphology import (
    calculate_volume,
    calculate_diameters_and_axes,
    calculate_sphericity,
    calculate_spiculation_index
)
from src.density import decompose_density, classify_nodule_type


class TestMorphology(unittest.TestCase):

    def setUp(self):
        # Create a 3D spherical test mask (radius = 5 voxels) in a 30x30x30 grid
        self.shape = (30, 30, 30)
        self.sphere_mask = np.zeros(self.shape, dtype=np.uint8)
        cz, cy, cx = 15, 15, 15
        r = 5
        z, y, x = np.ogrid[:30, :30, :30]
        dist_sq = (z - cz)**2 + (y - cy)**2 + (x - cx)**2
        self.sphere_mask[dist_sq <= r**2] = 1

    def test_volume_calculation(self):
        # Spacing (1, 1, 1) mm
        vol_iso = calculate_volume(self.sphere_mask, (1.0, 1.0, 1.0))
        active_count = np.sum(self.sphere_mask)
        self.assertEqual(vol_iso, float(active_count))

        # Spacing (2.0, 0.5, 0.5) mm -> voxel volume = 0.5 mm^3
        vol_aniso = calculate_volume(self.sphere_mask, (2.0, 0.5, 0.5))
        self.assertAlmostEqual(vol_aniso, active_count * 0.5, places=4)

    def test_diameters_and_axes(self):
        res = calculate_diameters_and_axes(self.sphere_mask, (1.0, 1.0, 1.0))
        self.assertIn("d_long_mm", res)
        self.assertIn("d_short_mm", res)
        self.assertIn("d_mean_mm", res)
        self.assertIn("principal_axes_mm", res)

        # For radius 5, diameter is ~10 mm
        self.assertGreaterEqual(res["d_mean_mm"], 8.0)
        self.assertLessEqual(res["d_mean_mm"], 12.0)
        # Symmetrical sphere should have close long and short diameters
        self.assertLess(abs(res["d_long_mm"] - res["d_short_mm"]), 2.0)

    def test_sphericity(self):
        psi = calculate_sphericity(self.sphere_mask, (1.0, 1.0, 1.0))
        # Sphere on grid should have high sphericity (> 0.85)
        self.assertGreater(psi, 0.85)
        self.assertLessEqual(psi, 1.0)

        # Elongated rod/cylinder should have significantly lower sphericity
        rod = np.zeros(self.shape, dtype=np.uint8)
        rod[14:16, 14:16, 5:25] = 1
        psi_rod = calculate_sphericity(rod, (1.0, 1.0, 1.0))
        self.assertLess(psi_rod, 0.70)
        self.assertLess(psi_rod, psi - 0.15)

    def test_spiculation_index(self):
        # Smooth sphere
        spic_sphere = calculate_spiculation_index(self.sphere_mask, (1.0, 1.0, 1.0))

        # Stellate shape with spicules radiating out
        stellate = self.sphere_mask.copy()
        stellate[15, 15, 0:30] = 1
        stellate[15, 0:30, 15] = 1
        stellate[0:30, 15, 15] = 1
        spic_stellate = calculate_spiculation_index(stellate, (1.0, 1.0, 1.0))

        self.assertGreater(spic_stellate, spic_sphere)


class TestDensityDecomposition(unittest.TestCase):

    def setUp(self):
        self.shape = (20, 20, 20)
        self.mask = np.ones(self.shape, dtype=np.uint8)

    def test_pure_ggn(self):
        # All voxels in GGO range (-500 HU)
        hu = np.full(self.shape, -500.0)
        res = decompose_density(hu, self.mask)
        self.assertEqual(res["nodule_type"], "pure_ggn")
        self.assertEqual(res["solid_ratio"], 0.0)
        self.assertEqual(res["ggo_ratio"], 1.0)

    def test_solid_nodule(self):
        # All voxels in soft tissue solid range (+20 HU)
        hu = np.full(self.shape, 20.0)
        res = decompose_density(hu, self.mask)
        self.assertEqual(res["nodule_type"], "solid")
        self.assertEqual(res["solid_ratio"], 1.0)
        self.assertGreater(res["solid_core_diameter_mm"], 0.0)

    def test_part_solid_nodule(self):
        # Half GGO (-500 HU) and half solid core (0 HU)
        hu = np.full(self.shape, -500.0)
        hu[:10, :, :] = 0.0  # 50% solid core
        res = decompose_density(hu, self.mask)
        self.assertEqual(res["nodule_type"], "part_solid")
        self.assertAlmostEqual(res["solid_ratio"], 0.5, places=2)

    def test_calcified_nodule(self):
        # High attenuation (+400 HU)
        hu = np.full(self.shape, 400.0)
        res = decompose_density(hu, self.mask)
        self.assertEqual(res["nodule_type"], "calcified")
        self.assertEqual(res["calc_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
