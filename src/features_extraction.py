"""
Objective, Deterministic 3D Computer Vision & Quantitative Profiling Module.
Extracts 27+ quantitative physical, morphometric, and densitometric features
directly from 3D CT arrays and binary masks.

All features are validated via Pydantic to ensure type safety, physical constraints,
and zero reliance on subjective human semantic ratings.
"""

import numpy as np
import scipy.ndimage
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from skimage import measure


class MorphologyFeatures(BaseModel):
    """Pydantic validated quantitative 3D morphology and geometry."""
    volume_mm3: float = Field(..., ge=0.0, description="True physical 3D volume in mm³")
    surface_area_mm2: float = Field(..., ge=0.0, description="Marching cubes surface area in mm²")
    surface_to_volume_ratio: float = Field(..., ge=0.0, description="Surface area / volume ratio (mm⁻¹)")
    d_long_mm: float = Field(..., ge=0.0, description="Major transverse inertia axis diameter (mm)")
    d_short_mm: float = Field(..., ge=0.0, description="Minor transverse inertia axis diameter (mm)")
    d_mean_mm: float = Field(..., ge=0.0, description="Mean transverse diameter (mm)")
    elongation: float = Field(..., ge=0.0, le=1.0, description="IBSI elongation: sqrt(lambda2 / lambda1)")
    flatness: float = Field(..., ge=0.0, le=1.0, description="IBSI flatness: sqrt(lambda3 / lambda1)")
    sphericity: float = Field(..., ge=0.0, le=1.05, description="Wadell sphericity Psi: (pi^(1/3) * (6V)^(2/3)) / A")
    radial_variance: float = Field(..., ge=0.0, description="Variance of centroid-to-surface Euclidean radii")
    spiculation_index: float = Field(..., ge=0.0, description="Normalized radial boundary variance spiculation index")


class DensityFeatures(BaseModel):
    """Pydantic validated 4-tier Hounsfield Unit attenuation distribution."""
    mean_hu: float = Field(..., description="Mean attenuation within nodule mask (HU)")
    median_hu: float = Field(..., description="Median attenuation within nodule mask (HU)")
    std_hu: float = Field(..., ge=0.0, description="Standard deviation of attenuation (HU)")
    min_hu: float = Field(..., description="Minimum attenuation (HU)")
    max_hu: float = Field(..., description="Maximum attenuation (HU)")
    p10_hu: float = Field(..., description="10th percentile attenuation (HU)")
    p25_hu: float = Field(..., description="25th percentile attenuation (HU)")
    p75_hu: float = Field(..., description="75th percentile attenuation (HU)")
    p90_hu: float = Field(..., description="90th percentile attenuation (HU)")
    iqr_hu: float = Field(..., ge=0.0, description="Interquartile range (p75 - p25) (HU)")
    
    # Priority-based tissue partition
    air_volume_mm3: float = Field(default=0.0, ge=0.0)
    ggo_volume_mm3: float = Field(default=0.0, ge=0.0)
    solid_volume_mm3: float = Field(default=0.0, ge=0.0)
    calc_volume_mm3: float = Field(default=0.0, ge=0.0)
    solid_core_volume_mm3: float = Field(default=0.0, ge=0.0)
    
    air_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    ggo_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    solid_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    calc_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    solid_core_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    solid_core_diameter_mm: float = Field(default=0.0, ge=0.0)
    nodule_type: str = Field(default="solid", description="pure_ggn, part_solid, solid, or calcified")


class ClinicalContext(BaseModel):
    """Optional clinical covariates with standard clinical defaults."""
    age: float = Field(default=62.0, ge=18.0, le=120.0)
    is_female: bool = Field(default=False)
    is_upper_lobe: bool = Field(default=True)
    family_history: bool = Field(default=False)
    emphysema: bool = Field(default=False)


class NoduleFeatures(BaseModel):
    """Complete, autonomous 27+ feature representation for a pulmonary nodule."""
    morphology: MorphologyFeatures
    density: DensityFeatures
    clinical: ClinicalContext = Field(default_factory=ClinicalContext)

    def to_feature_vector(self) -> np.ndarray:
        """Converts into a deterministic numerical feature vector for ML inference."""
        m = self.morphology
        d = self.density
        c = self.clinical
        return np.array([
            m.volume_mm3,
            m.d_long_mm,
            m.d_short_mm,
            m.d_mean_mm,
            m.elongation,
            m.flatness,
            m.sphericity,
            m.surface_to_volume_ratio,
            m.radial_variance,
            m.spiculation_index,
            d.mean_hu,
            d.median_hu,
            d.std_hu,
            d.p10_hu,
            d.p25_hu,
            d.p75_hu,
            d.p90_hu,
            d.iqr_hu,
            d.ggo_ratio,
            d.solid_ratio,
            d.calc_ratio,
            d.solid_core_ratio,
            d.solid_core_diameter_mm,
            1.0 if d.nodule_type == "solid" else 2.0 if d.nodule_type == "part_solid" else 3.0 if d.nodule_type == "pure_ggn" else 4.0,
            c.age,
            1.0 if c.is_female else 0.0,
            1.0 if c.is_upper_lobe else 0.0,
            1.0 if c.family_history else 0.0,
            1.0 if c.emphysema else 0.0
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> list:
        return [
            "volume_mm3", "d_long_mm", "d_short_mm", "d_mean_mm", "elongation", "flatness",
            "sphericity", "surface_to_volume_ratio", "radial_variance", "spiculation_index",
            "mean_hu", "median_hu", "std_hu", "p10_hu", "p25_hu", "p75_hu", "p90_hu", "iqr_hu",
            "ggo_ratio", "solid_ratio", "calc_ratio", "solid_core_ratio", "solid_core_diameter_mm",
            "nodule_type_encoded", "age", "is_female", "is_upper_lobe", "family_history", "emphysema"
        ]


def extract_morphology_features(
    mask_3d: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> MorphologyFeatures:
    """
    Computes IBSI-compliant 3D morphology and surface curvature.
    """
    voxel_count = int(np.sum(mask_3d > 0))
    if voxel_count == 0:
        raise ValueError("Cannot extract morphology from an empty nodule mask.")

    sz, sy, sx = spacing
    voxel_vol = sz * sy * sx
    vol_mm3 = float(voxel_count * voxel_vol)

    # Marching cubes surface area & mesh
    try:
        padded = np.pad(mask_3d, 1, mode="constant", constant_values=0)
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing)
        surface_area = float(measure.mesh_surface_area(verts, faces))
    except Exception:
        # Spherical fallback
        r_equiv = (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        surface_area = float(4.0 * np.pi * (r_equiv ** 2))
        verts = None

    # Wadell Sphericity: Psi = (pi^(1/3) * (6V)^(2/3)) / A
    if surface_area > 0:
        sphericity = float((np.pi ** (1.0 / 3.0)) * ((6.0 * vol_mm3) ** (2.0 / 3.0)) / surface_area)
        sphericity = min(1.0, max(0.01, sphericity))
    else:
        sphericity = 1.0

    s2v = float(surface_area / max(1e-4, vol_mm3))

    # Second-moment inertia tensor in 3D
    coords = np.argwhere(mask_3d > 0).astype(np.float64)
    # Physical coordinates
    phys_coords = coords * np.array([sz, sy, sx])
    centroid = np.mean(phys_coords, axis=0)
    centered = phys_coords - centroid

    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 2:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        eigvals = np.sort(eigvals)[::-1]  # lambda1 >= lambda2 >= lambda3
        l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]
        
        # Radii of equivalent ellipsoid: r_i = 2 * sqrt(lambda_i)
        d_long = float(4.0 * np.sqrt(l1))
        d_short = float(4.0 * np.sqrt(l2))
        d_mean = (d_long + d_short) / 2.0

        elongation = float(np.sqrt(l2 / l1))
        flatness = float(np.sqrt(l3 / l1))
    else:
        d_long = d_short = d_mean = float(2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))
        elongation = 1.0
        flatness = 1.0

    # 3D Radial boundary variance & spiculation
    if verts is not None and len(verts) > 10:
        # Distance from mesh vertices to center of mass
        mesh_center = np.mean(verts, axis=0)
        radii = np.linalg.norm(verts - mesh_center, axis=1)
        mean_r = np.mean(radii)
        rad_var = float(np.var(radii))
        spic_idx = float(np.std(radii) / max(1e-3, mean_r))
    else:
        rad_var = 0.0
        spic_idx = 0.0

    return MorphologyFeatures(
        volume_mm3=round(vol_mm3, 2),
        surface_area_mm2=round(surface_area, 2),
        surface_to_volume_ratio=round(s2v, 4),
        d_long_mm=round(d_long, 2),
        d_short_mm=round(d_short, 2),
        d_mean_mm=round(d_mean, 2),
        elongation=round(elongation, 4),
        flatness=round(flatness, 4),
        sphericity=round(sphericity, 4),
        radial_variance=round(rad_var, 4),
        spiculation_index=round(spic_idx, 4)
    )


def extract_density_features(
    ct_volume_hu: np.ndarray,
    mask_3d: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> DensityFeatures:
    """
    Computes complete 4-tier HU distribution, percentiles, and tissue decomposition.
    """
    nodule_hu = ct_volume_hu[mask_3d > 0]
    if len(nodule_hu) == 0:
        raise ValueError("Cannot extract densitometry from an empty nodule mask.")

    sz, sy, sx = spacing
    voxel_vol = sz * sy * sx
    total_vol = float(len(nodule_hu) * voxel_vol)

    # Attenuation statistics
    mean_hu = float(np.mean(nodule_hu))
    median_hu = float(np.median(nodule_hu))
    std_hu = float(np.std(nodule_hu))
    min_hu = float(np.min(nodule_hu))
    max_hu = float(np.max(nodule_hu))

    p10 = float(np.percentile(nodule_hu, 10))
    p25 = float(np.percentile(nodule_hu, 25))
    p75 = float(np.percentile(nodule_hu, 75))
    p90 = float(np.percentile(nodule_hu, 90))
    iqr_hu = float(p75 - p25)

    # 4-tier HU decomposition
    air_count = int(np.sum(nodule_hu < -750.0))
    ggo_count = int(np.sum((nodule_hu >= -750.0) & (nodule_hu < -300.0)))
    solid_count = int(np.sum((nodule_hu >= -300.0) & (nodule_hu <= 200.0)))
    calc_count = int(np.sum(nodule_hu > 200.0))
    solid_core_count = int(np.sum(nodule_hu >= -160.0))

    air_vol = float(air_count * voxel_vol)
    ggo_vol = float(ggo_count * voxel_vol)
    solid_vol = float(solid_count * voxel_vol)
    calc_vol = float(calc_count * voxel_vol)
    solid_core_vol = float(solid_core_count * voxel_vol)

    air_ratio = air_vol / max(1e-4, total_vol)
    ggo_ratio = ggo_vol / max(1e-4, total_vol)
    solid_ratio = solid_vol / max(1e-4, total_vol)
    calc_ratio = calc_vol / max(1e-4, total_vol)
    solid_core_ratio = solid_core_vol / max(1e-4, total_vol)

    if solid_core_vol > 0:
        solid_core_diam = float(2.0 * ((3.0 * solid_core_vol) / (4.0 * np.pi)) ** (1.0 / 3.0))
    else:
        solid_core_diam = 0.0

    # Nodule category
    if calc_ratio >= 0.50:
        nod_type = "calcified"
    elif solid_ratio < 0.10:
        nod_type = "pure_ggn"
    elif solid_ratio >= 0.80:
        nod_type = "solid"
    else:
        nod_type = "part_solid"

    return DensityFeatures(
        mean_hu=round(mean_hu, 1),
        median_hu=round(median_hu, 1),
        std_hu=round(std_hu, 1),
        min_hu=round(min_hu, 1),
        max_hu=round(max_hu, 1),
        p10_hu=round(p10, 1),
        p25_hu=round(p25, 1),
        p75_hu=round(p75, 1),
        p90_hu=round(p90, 1),
        iqr_hu=round(iqr_hu, 1),
        air_volume_mm3=round(air_vol, 2),
        ggo_volume_mm3=round(ggo_vol, 2),
        solid_volume_mm3=round(solid_vol, 2),
        calc_volume_mm3=round(calc_vol, 2),
        solid_core_volume_mm3=round(solid_core_vol, 2),
        air_ratio=round(air_ratio, 4),
        ggo_ratio=round(ggo_ratio, 4),
        solid_ratio=round(solid_ratio, 4),
        calc_ratio=round(calc_ratio, 4),
        solid_core_ratio=round(solid_core_ratio, 4),
        solid_core_diameter_mm=round(solid_core_diam, 2),
        nodule_type=nod_type
    )


def extract_nodule_features(
    ct_volume_hu: np.ndarray,
    mask_3d: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    clinical: Optional[ClinicalContext] = None
) -> NoduleFeatures:
    """
    Unified entry point: Extracts all 27+ deterministic quantitative features
    directly from 3D CT volume and mask.
    """
    morphology = extract_morphology_features(mask_3d, spacing)
    density = extract_density_features(ct_volume_hu, mask_3d, spacing)
    clin = clinical or ClinicalContext()
    return NoduleFeatures(morphology=morphology, density=density, clinical=clin)
