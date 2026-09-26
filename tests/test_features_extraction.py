"""
Unit tests for deterministic 3D computer vision and densitometry extraction (src/features_extraction.py).
"""

import numpy as np
import pytest
from src.features_extraction import (
    extract_morphology_features,
    extract_density_features,
    extract_nodule_features,
    NoduleFeatures,
    MorphologyFeatures,
    DensityFeatures,
    ClinicalContext
)


def test_morphology_extraction_synthetic_sphere():
    shape = (32, 32, 32)
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = 16, 16, 16
    z, y, x = np.ogrid[:32, :32, :32]
    r = 6.0
    mask[(z - cz)**2 + (y - cy)**2 + (x - cx)**2 <= r**2] = 1

    morph = extract_morphology_features(mask, spacing=(1.0, 1.0, 1.0))
    assert isinstance(morph, MorphologyFeatures)
    assert morph.volume_mm3 > 700.0 and morph.volume_mm3 < 1100.0
    assert morph.sphericity > 0.80  # Spheres should have high sphericity
    assert morph.d_long_mm >= morph.d_short_mm
    assert morph.elongation > 0.85
    assert morph.flatness > 0.85


def test_density_extraction_part_solid():
    shape = (32, 32, 32)
    mask = np.zeros(shape, dtype=np.uint8)
    ct = np.full(shape, -900.0, dtype=np.float32)

    cz, cy, cx = 16, 16, 16
    z, y, x = np.ogrid[:32, :32, :32]
    dist_sq = (z - cz)**2 + (y - cy)**2 + (x - cx)**2

    # Outer GGO (dist <= 8)
    mask[dist_sq <= 8.0**2] = 1
    ct[dist_sq <= 8.0**2] = -500.0

    # Inner Solid core (dist <= 4)
    ct[dist_sq <= 4.0**2] = 50.0

    dens = extract_density_features(ct, mask, spacing=(1.0, 1.0, 1.0))
    assert isinstance(dens, DensityFeatures)
    assert dens.nodule_type == "part_solid"
    assert dens.solid_ratio > 0.05 and dens.solid_ratio < 0.60
    assert dens.ggo_ratio > 0.40
    assert dens.solid_core_diameter_mm > 5.0
    assert dens.mean_hu < 0.0  # Overall negative due to GGO envelope


def test_nodule_features_pydantic_vector():
    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:15, 5:15, 5:15] = 1
    ct = np.full(shape, -100.0, dtype=np.float32)

    nodule_feats = extract_nodule_features(
        ct_volume_hu=ct,
        mask_3d=mask,
        spacing=(1.0, 1.0, 1.0),
        clinical=ClinicalContext(age=70.0, is_female=True, is_upper_lobe=True)
    )

    assert isinstance(nodule_feats, NoduleFeatures)
    vec = nodule_feats.to_feature_vector()
    assert isinstance(vec, np.ndarray)
    assert len(vec) == len(NoduleFeatures.feature_names())
    assert vec[0] == nodule_feats.morphology.volume_mm3
