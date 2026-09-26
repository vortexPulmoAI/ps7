"""
Radiographic attenuation and density decomposition for pulmonary nodules.
Segments nodule voxels into Air, Ground-Glass Opacity (GGO), Solid Soft-Tissue,
and Macrocalcification, classifying lesions into Pure GGN, Part-Solid, Solid, or Calcified.
"""

import numpy as np


def decompose_density(volume_hu, mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Decomposes the internal attenuation profile of a segmented nodule into
    established clinical density compartments:
        - Air / Aerated Entrapment: < -750 HU
        - Ground-Glass Opacity (GGO): [-750, -300 HU]
        - Solid Component: > -300 HU
        - Solid Core (dense soft-tissue): >= -160 HU
        - Macrocalcification: > +200 HU

    Args:
        volume_hu: 2D or 3D numpy array in calibrated Hounsfield Units
        mask: binary array of the nodule segmentation
        voxel_spacing: (sz, sy, sx) in mm

    Returns:
        dict containing volume breakdowns, percentage ratios, and nodule subtype
    """
    mask_bool = mask > 0
    nodule_hu = volume_hu[mask_bool]
    total_voxels = len(nodule_hu)

    unit_vol = float(np.prod(voxel_spacing))

    if total_voxels == 0:
        return {
            "total_volume_mm3": 0.0,
            "air_volume_mm3": 0.0,
            "ggo_volume_mm3": 0.0,
            "solid_volume_mm3": 0.0,
            "solid_core_volume_mm3": 0.0,
            "calc_volume_mm3": 0.0,
            "solid_ratio": 0.0,
            "solid_core_ratio": 0.0,
            "ggo_ratio": 0.0,
            "calc_ratio": 0.0,
            "nodule_type": "unknown",
            "mean_hu": 0.0,
            "solid_core_diameter_mm": 0.0,
        }

    # Count voxel categories
    air_count = int(np.count_nonzero(nodule_hu < -750.0))
    ggo_count = int(np.count_nonzero((nodule_hu >= -750.0) & (nodule_hu <= -300.0)))
    solid_count = int(np.count_nonzero(nodule_hu > -300.0))
    solid_core_count = int(np.count_nonzero(nodule_hu >= -160.0))
    calc_count = int(np.count_nonzero(nodule_hu > 200.0))

    # Calculate physical volumes in mm^3
    total_vol = total_voxels * unit_vol
    air_vol = air_count * unit_vol
    ggo_vol = ggo_count * unit_vol
    solid_vol = solid_count * unit_vol
    solid_core_vol = solid_core_count * unit_vol
    calc_vol = calc_count * unit_vol

    # Ratios
    solid_ratio = solid_vol / total_vol
    solid_core_ratio = solid_core_vol / total_vol
    ggo_ratio = ggo_vol / total_vol
    calc_ratio = calc_vol / total_vol

    # Estimate solid core equivalent diameter (spherical approximation d = 2 * (3V / 4pi)^(1/3))
    if solid_core_vol > 0.0:
        solid_core_diameter = 2.0 * ((3.0 * solid_core_vol) / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        solid_core_diameter = 0.0

    nodule_type = classify_nodule_type(solid_ratio, ggo_ratio, calc_ratio, solid_core_ratio)

    return {
        "total_volume_mm3": round(total_vol, 2),
        "air_volume_mm3": round(air_vol, 2),
        "ggo_volume_mm3": round(ggo_vol, 2),
        "solid_volume_mm3": round(solid_vol, 2),
        "solid_core_volume_mm3": round(solid_core_vol, 2),
        "calc_volume_mm3": round(calc_vol, 2),
        "solid_ratio": round(solid_ratio, 4),
        "solid_core_ratio": round(solid_core_ratio, 4),
        "ggo_ratio": round(ggo_ratio, 4),
        "calc_ratio": round(calc_ratio, 4),
        "nodule_type": nodule_type,
        "mean_hu": round(float(np.mean(nodule_hu)), 1),
        "solid_core_diameter_mm": round(float(solid_core_diameter), 1),
    }


def classify_nodule_type(solid_ratio, ggo_ratio, calc_ratio, solid_core_ratio=None):
    """
    Classifies nodule attenuation subtype in accordance with Fleischner / Lung-RADS criteria:
        - "calcified": calc_ratio >= 0.50 (benign pattern)
        - "pure_ggn": solid_ratio < 0.10 (non-solid, ground-glass opacity)
        - "solid": solid_ratio >= 0.80 (predominantly soft-tissue attenuation)
        - "part_solid": contains both ground glass and internal solid core (0.10 <= solid_ratio < 0.80)
    """
    if calc_ratio >= 0.50:
        return "calcified"
    if solid_ratio < 0.10:
        return "pure_ggn"
    if solid_ratio >= 0.80:
        return "solid"
    return "part_solid"
