"""
End-to-End Pipeline Demonstration Script for Pulmonary Nodule Analysis.
Demonstrates:
1. Real dataset parsing from C:/PARTH/ps7/dataset (LUNA16 annotations/candidates, LIDC XMLs, lung masks)
2. CT raw slice ingestion (using local 124.raw) and lung windowing
3. Volumetric morphological and density feature extraction
4. Clinical risk estimation via Brock / PanCan model and ACR Lung-RADS Version 2022
"""

import os
import numpy as np
from src.preprocessing import apply_hu_window, AffineTransform
from src.morphology import (
    calculate_volume,
    calculate_diameters_and_axes,
    calculate_sphericity,
    calculate_spiculation_index
)
from src.density import decompose_density
from src.dataset_parser import (
    LUNA16AnnotationParser,
    LUNA16CandidateParser,
    LIDCXMLParser,
    LUNALungMaskLoader
)
from src.detection import NNDetectionDataPrep, CandidatePatchExtractor, FROCEvaluator
from src.segmentation import NNUNetDataPrep, SegmentationEvaluator
from src.risk_engine import BrockCalculator, LungRADSClassifier, assess_risk


import sys

# Configure UTF-8 stdout if available
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_pipeline_demo():
    print("=" * 80)
    print("[*] PULMONARY NODULE RISK ASSESSMENT AI -- PIPELINE DEMO")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")

    # -------------------------------------------------------------
    # 1. Dataset Verification
    # -------------------------------------------------------------
    print("\n[1/4] Verifying Real Datasets in 'dataset/'...")
    annotations_path = os.path.join(dataset_dir, "annotations.csv")
    candidates_path = os.path.join(dataset_dir, "candidates.csv")
    xml_root = os.path.join(dataset_dir, "LIDC-XML-only", "tcia-lidc-xml")
    lungs_dir = os.path.join(dataset_dir, "seg-lungs-LUNA16")

    # LUNA16 Annotations
    anno_parser = LUNA16AnnotationParser(annotations_path)
    total_annotations = len(anno_parser)
    print(f"  ✓ LUNA16 Annotations: {total_annotations:,} consensus nodules loaded.")
    sample_nodule = anno_parser.get_all()[0]
    print(f"    Sample Nodule: SeriesUID={sample_nodule['seriesuid'][:28]}... "
          f"Coords=({sample_nodule['coord_x']:.1f}, {sample_nodule['coord_y']:.1f}, {sample_nodule['coord_z']:.1f}) mm, "
          f"Diameter={sample_nodule['diameter_mm']:.1f} mm")

    # LUNA16 Candidates
    cand_parser = LUNA16CandidateParser(candidates_path)
    cand_sample = cand_parser.get_sample_candidates(max_rows=5)
    print(f"  ✓ LUNA16 Candidates: Streamed sample of {len(cand_sample)} candidate records.")

    # LIDC XML Annotations
    xml_parser = LIDCXMLParser()
    sample_xml = os.path.join(xml_root, "157", "158.xml")
    if os.path.exists(sample_xml):
        xml_data = xml_parser.parse_file(sample_xml)
        print(f"  ✓ LIDC XML Parsed: {len(xml_data['nodules'])} nodules in series {xml_data['series_uid'][:28]}...")
        for i, nod in enumerate(xml_data["nodules"]):
            chars = nod["characteristics"]
            if chars:
                print(f"    Nodule {nod['nodule_id']}: Malignancy Score={chars.get('malignancy')}, "
                      f"Spiculation={chars.get('spiculation')}, Option B Label={nod['option_b_label']} "
                      f"({ 'Benign (0)' if nod['option_b_label']==0 else 'Malignant (1)' if nod['option_b_label']==1 else 'Dropped (Ambiguous 3)' })")

    # LUNA16 Lung Mask
    mask_loader = LUNALungMaskLoader(lungs_dir)
    sample_series = sample_nodule["seriesuid"]
    mhd_found = mask_loader.find_mhd_for_series(sample_series)
    if mhd_found:
        print(f"  ✓ LUNA16 Lung Mask: Found matching MHD header for series {sample_series[:28]}...")

    # -------------------------------------------------------------
    # 2. Local CT Slice Ingestion & Windowing (124.raw)
    # -------------------------------------------------------------
    print("\n[2/4] Ingesting CT Slice (124.raw) & Radiometric Normalization...")
    raw_path = os.path.join(base_dir, "124.raw")
    if os.path.exists(raw_path):
        slice_hu = np.fromfile(raw_path, dtype=np.int16).reshape((512, 512))
        print(f"  ✓ Loaded 124.raw: shape={slice_hu.shape}, min HU={slice_hu.min()}, max HU={slice_hu.max()}")
        lung_window = apply_hu_window(slice_hu, window="lung", normalize=True)
        soft_tissue = apply_hu_window(slice_hu, window="soft_tissue", normalize=True)
        print(f"  ✓ Applied Lung Window (WL: -600, WW: 1500) -> range [{lung_window.min():.2f}, {lung_window.max():.2f}]")
        print(f"  ✓ Applied Soft Tissue Window (WL: 40, WW: 400) -> range [{soft_tissue.min():.2f}, {soft_tissue.max():.2f}]")

    # -------------------------------------------------------------
    # 3. 3D Volumetric Morphology & Density Profiling
    # -------------------------------------------------------------
    print("\n[3/4] Extracting Quantitative 3D Morphology & Density Profiles...")
    # Construct a synthetic 3D test nodule (Part-Solid with spiculated margin)
    shape_3d = (40, 40, 40)
    nodule_mask_3d = np.zeros(shape_3d, dtype=np.uint8)
    ct_volume_hu = np.full(shape_3d, -900.0, dtype=np.float32)  # surrounding parenchyma

    cz, cy, cx = 20, 20, 20
    z, y, x = np.ogrid[:40, :40, :40]
    r_core = 4.5
    r_outer = 8.0
    dist_sq = (z - cz)**2 + (y - cy)**2 + (x - cx)**2

    # Outer GGO envelope
    nodule_mask_3d[dist_sq <= r_outer**2] = 1
    ct_volume_hu[dist_sq <= r_outer**2] = -500.0  # GGO attenuation

    # Inner Solid core
    ct_volume_hu[dist_sq <= r_core**2] = 20.0  # Solid soft-tissue attenuation

    # Add 4 radial spiculations
    nodule_mask_3d[cz, cy-12:cy+13, cx] = 1
    ct_volume_hu[cz, cy-12:cy+13, cx] = -200.0
    nodule_mask_3d[cz, cy, cx-12:cx+13] = 1
    ct_volume_hu[cz, cy, cx-12:cx+13] = -200.0

    spacing = (1.0, 1.0, 1.0)
    vol_mm3 = calculate_volume(nodule_mask_3d, spacing)
    axes = calculate_diameters_and_axes(nodule_mask_3d, spacing)
    sphericity = calculate_sphericity(nodule_mask_3d, spacing)
    spiculation_idx = calculate_spiculation_index(nodule_mask_3d, spacing)
    density_profile = decompose_density(ct_volume_hu, nodule_mask_3d, spacing)

    print(f"  ✓ Physical Volume:       {vol_mm3:.1f} mm³")
    print(f"  ✓ Transverse Diameters:  d_long={axes['d_long_mm']:.1f} mm, d_short={axes['d_short_mm']:.1f} mm, d_mean={axes['d_mean_mm']:.1f} mm")
    print(f"  ✓ Sphericity (Ψ):        {sphericity:.3f} (1.0 = smooth sphere)")
    print(f"  ✓ Spiculation Index:     {spiculation_idx:.4f} (elevated boundary variance)")
    print(f"  ✓ Attenuation Subtype:   {density_profile['nodule_type'].upper()}")
    print(f"    - Solid Core Volume:   {density_profile['solid_core_volume_mm3']:.1f} mm³ ({density_profile['solid_core_ratio']*100:.1f}%)")
    print(f"    - Solid Core Diameter: {density_profile['solid_core_diameter_mm']:.1f} mm")
    print(f"    - GGO Volume:          {density_profile['ggo_volume_mm3']:.1f} mm³ ({density_profile['ggo_ratio']*100:.1f}%)")

    # -------------------------------------------------------------
    # 4. Clinical Triad Malignancy Risk Engine (assess_risk)
    # -------------------------------------------------------------
    print("\n[4/4] Evaluating Clinical Risk Models & Calibrated Malignancy Engine...")
    nodule_features = {
        "mean_diameter_mm": axes["d_mean_mm"],
        "d_long_mm": axes["d_long_mm"],
        "nodule_type": density_profile["nodule_type"],
        "solid_core_diameter_mm": density_profile["solid_core_diameter_mm"],
        "is_spiculated": True,
        "is_upper_lobe": True,
        "sphericity": sphericity,
        "spiculation": 4.0,
        "subtlety": 4.0,
        "margin": 3.0,
        "lobulation": 3.0,
        "texture": 5.0,
        "calcification": 6.0,
        "roi_count": 8.0,
        "age": 65.0,
        "is_female": True,
        "family_history": False,
        "emphysema": True,
    }

    triad_report = assess_risk(nodule_features)

    print("\n" + "=" * 80)
    print("CLINICAL NODULE RISK ASSESSMENT REPORT")
    print("=" * 80)
    print(f"* Lesion Classification:   {density_profile['nodule_type'].replace('_', ' ').title()}")
    print(f"* Mean Transverse Size:    {axes['d_mean_mm']:.1f} mm (Max: {axes['d_long_mm']:.1f} mm)")
    print(f"* Solid Core Size:         {density_profile['solid_core_diameter_mm']:.1f} mm")
    print(f"* Sphericity / Spiculation: Psi={sphericity:.2f} | Spiculation Index={spiculation_idx:.4f}")
    print("-" * 80)
    print(f"* 1. Calibrated ML Risk:   {triad_report['risk_percent']:.1f}% Malignancy Probability (Trained on LIDC-IDRI)")
    print(f"* 2. Brock / PanCan Score: {triad_report['brock_percent']:.1f}% Reference Probability")
    print(f"     Guideline Action:     {triad_report['bts_recommendation']}")
    print(f"* 3. ACR Lung-RADS v2022:  Category {triad_report['lung_rads_category']}")
    print(f"     Clinical Management:  {triad_report['lung_rads_management']}")
    print("-" * 80)
    print(f"* Concordance Status:      {triad_report['confidence_flag']}")
    print("=" * 80 + "\n")

    # Step 6: Standardized Diagnostic JSON Output
    import json
    from src.risk_engine import generate_diagnostic_json
    diagnostic_json = generate_diagnostic_json(
        candidate_id="NOD_0042",
        centroid_world_mm=(-64.2, 42.8, -180.5),
        dimensions={
            "volume_mm3": vol_mm3,
            "d_long_mm": axes["d_long_mm"],
            "d_short_mm": axes["d_short_mm"],
            "d_mean_mm": axes["d_mean_mm"],
        },
        morphology={
            "type": density_profile["nodule_type"],
            "solid_core_diameter_mm": density_profile["solid_core_diameter_mm"],
            "is_spiculated": True,
        },
        risk_assessment=triad_report,
    )
    print("[*] STANDARDIZED DIAGNOSTIC JSON PAYLOAD (Step 6 Specification):")
    print(json.dumps(diagnostic_json, indent=2))
    print()


if __name__ == "__main__":
    run_pipeline_demo()
