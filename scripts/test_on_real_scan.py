"""
End-to-End Real Patient Scan Testing CLI for Pulmonary Nodule Analysis.
Loads a real 3D CT scan from LUNA16 subset0, extracts the verified nodule,
performs 3D morphology and densitometric profiling, evaluates the trained
calibrated malignancy model, Brock formula, and ACR Lung-RADS v2022,
and outputs both a clinical report and the standardized Step 6 JSON payload.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.luna_dataset import LUNA16SubsetIngestor
from src.preprocessing import resample_isotropic
from src.morphology import (
    calculate_volume,
    calculate_diameters_and_axes,
    calculate_sphericity,
    calculate_spiculation_index
)
from src.density import decompose_density
from src.risk_engine import (
    CalibratedMalignancyModel,
    BrockCalculator,
    LungRADSClassifier,
    assess_risk,
    generate_diagnostic_json
)


def segment_patch_nodule(patch_hu: np.ndarray, expected_radius_vox: float) -> np.ndarray:
    """
    Segment the nodule within the normalized isotropic 3D patch using
    attenuation gating and radial distance weighting around patch center.
    """
    cz, cy, cx = [s // 2 for s in patch_hu.shape]
    z, y, x = np.ogrid[:patch_hu.shape[0], :patch_hu.shape[1], :patch_hu.shape[2]]
    dist = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)

    # Nodule attenuation mask: parenchyma is typically < -700 HU; nodules are [-600, +200]
    attenuation_mask = (patch_hu >= -650.0) & (patch_hu <= 350.0)

    # Spatial region of interest based on known consensus diameter with safety margin
    roi_mask = dist <= max(expected_radius_vox * 1.35, 3.0)

    nodule_mask = (attenuation_mask & roi_mask).astype(np.uint8)

    # Fallback to sphere if boundary is diffuse
    if nodule_mask.sum() < 8:
        nodule_mask = (dist <= max(expected_radius_vox, 2.0)).astype(np.uint8)

    return nodule_mask


def run_real_scan_test(scan_idx: int = 0, series_uid: str = None, save_viz: bool = True):
    print("=" * 80)
    print("[*] PULMONARY NODULE RISK ASSESSMENT -- REAL SCAN INFERENCE")
    print("=" * 80)

    ingestor = LUNA16SubsetIngestor(dataset_root="dataset")
    scans = ingestor.discover_subset_scans("subset0")

    matching_scans = []
    for s in scans:
        uid = os.path.splitext(os.path.basename(s))[0]
        nodules = ingestor.anno_parser.get_by_series(uid)
        if nodules:
            matching_scans.append((s, nodules))

    print(f"[*] Available subset0 scans with consensus nodules: {len(matching_scans)}")
    if not matching_scans:
        print("[!] No annotated scans found in subset0.")
        return

    # Select scan
    if series_uid:
        selected_scan = None
        selected_nodules = None
        for s, nod in matching_scans:
            if series_uid in s:
                selected_scan = s
                selected_nodules = nod
                break
        if not selected_scan:
            print(f"[!] Scan with SeriesUID matching '{series_uid}' not found. Defaulting to index 0.")
            selected_scan, selected_nodules = matching_scans[0]
    else:
        scan_idx = max(0, min(scan_idx, len(matching_scans) - 1))
        selected_scan, selected_nodules = matching_scans[scan_idx]

    uid = os.path.splitext(os.path.basename(selected_scan))[0]
    print(f"\n[+] Selected Scan #{scan_idx}: {uid}")
    print(f"    Total Confirmed Nodule(s) in Scan: {len(selected_nodules)}")

    # Load 3D Scan
    print(f"[*] Loading 3D Volume and paired lung mask...")
    scan_data = ingestor.load_scan(selected_scan)
    vol_hu = scan_data["volume_hu"]
    spacing = scan_data["spacing_zyx"]
    origin = scan_data["origin_xyz"]
    lung_mask = scan_data["lung_mask"]

    print(f"    CT Dimensions:    {vol_hu.shape} (slices, rows, cols)")
    print(f"    Original Spacing: {spacing} (sz, sy, sx) mm")
    print(f"    HU Range:         [{vol_hu.min()}, {vol_hu.max()}] HU")

    # Resample entire scan or crop locally
    # To be fast and memory-efficient, resample a subvolume around each nodule
    for nodule_i, nod in enumerate(selected_nodules, 1):
        world_coord = np.array([nod["coord_x"], nod["coord_y"], nod["coord_z"]])
        diameter_mm = nod["diameter_mm"]
        expected_r_vox = (diameter_mm / 2.0)

        print("\n" + "-" * 80)
        print(f"[*] ANALYZING NODULE #{nodule_i} OF {len(selected_nodules)}")
        print(f"    World Centroid: ({world_coord[0]:.2f}, {world_coord[1]:.2f}, {world_coord[2]:.2f}) mm")
        print(f"    LUNA16 Consensus Diameter: {diameter_mm:.2f} mm")

        # Convert world coord to voxel index in original scan
        # origin is (x, y, z), spacing is (sz, sy, sx)
        vox_x = int(round((world_coord[0] - origin[0]) / spacing[2]))
        vox_y = int(round((world_coord[1] - origin[1]) / spacing[1]))
        vox_z = int(round((world_coord[2] - origin[2]) / spacing[0]))

        print(f"    Scan Voxel Coordinates: (z={vox_z}, y={vox_y}, x={vox_x})")

        # Crop a localized 3D subvolume around nodule with padding
        # Margin: +/- 30mm around nodule
        margin_z = int(np.ceil(30.0 / spacing[0]))
        margin_y = int(np.ceil(30.0 / spacing[1]))
        margin_x = int(np.ceil(30.0 / spacing[2]))

        z_min, z_max = max(0, vox_z - margin_z), min(vol_hu.shape[0], vox_z + margin_z + 1)
        y_min, y_max = max(0, vox_y - margin_y), min(vol_hu.shape[1], vox_y + margin_y + 1)
        x_min, x_max = max(0, vox_x - margin_x), min(vol_hu.shape[2], vox_x + margin_x + 1)

        subvol_hu = vol_hu[z_min:z_max, y_min:y_max, x_min:x_max]

        # Resample local subvolume to 1.0mm isotropic
        subvol_iso, iso_spacing = resample_isotropic(
            subvol_hu,
            current_spacing=spacing,
            target_spacing=(1.0, 1.0, 1.0),
            is_mask=False
        )

        # Center in isotropic patch
        iso_cz = int(round((vox_z - z_min) * (spacing[0] / 1.0)))
        iso_cy = int(round((vox_y - y_min) * (spacing[1] / 1.0)))
        iso_cx = int(round((vox_x - x_min) * (spacing[2] / 1.0)))

        patch_rad = 24  # 48x48x48 mm bounding box
        pz_min, pz_max = max(0, iso_cz - patch_rad), min(subvol_iso.shape[0], iso_cz + patch_rad)
        py_min, py_max = max(0, iso_cy - patch_rad), min(subvol_iso.shape[1], iso_cy + patch_rad)
        px_min, px_max = max(0, iso_cx - patch_rad), min(subvol_iso.shape[2], iso_cx + patch_rad)

        patch_hu = subvol_iso[pz_min:pz_max, py_min:py_max, px_min:px_max]

        # Segment nodule in 3D patch
        nodule_mask_3d = segment_patch_nodule(patch_hu, expected_radius_vox=expected_r_vox)

        # Quantitative 3D Morphology
        target_iso = (1.0, 1.0, 1.0)
        vol_mm3 = calculate_volume(nodule_mask_3d, target_iso)
        axes = calculate_diameters_and_axes(nodule_mask_3d, target_iso)
        sphericity = calculate_sphericity(nodule_mask_3d, target_iso)
        spiculation_idx = calculate_spiculation_index(nodule_mask_3d, target_iso)
        is_spic = spiculation_idx > 0.05

        # 4-Tier Density Profiling
        density_prof = decompose_density(patch_hu, nodule_mask_3d, target_iso)

        # Determine anatomical lobe (approximate from Z coordinate)
        is_upper_lobe = world_coord[2] > -160.0

        # Run Clinical Risk Triad Engine
        features = {
            "mean_diameter_mm": axes["d_mean_mm"],
            "d_long_mm": axes["d_long_mm"],
            "nodule_type": density_prof["nodule_type"],
            "solid_core_diameter_mm": density_prof["solid_core_diameter_mm"],
            "is_spiculated": is_spic,
            "is_upper_lobe": is_upper_lobe,
            "sphericity": sphericity,
            "spiculation": min(5.0, max(1.0, 1.0 + spiculation_idx * 40.0)),
            "subtlety": 4.0,
            "margin": 3.0,
            "lobulation": 2.0,
            "texture": 5.0 if density_prof["nodule_type"] == "solid" else 3.0,
            "calcification": 6.0,
            "roi_count": 4.0,
            "age": 62.0,
            "is_female": False,
            "family_history": False,
            "emphysema": False,
        }

        triad_report = assess_risk(features)

        # Generate Standardized Step 6 JSON
        diag_json = generate_diagnostic_json(
            candidate_id=f"LUNA_{uid[:8]}_N{nodule_i:02d}",
            centroid_world_mm=(float(world_coord[0]), float(world_coord[1]), float(world_coord[2])),
            dimensions={
                "volume_mm3": vol_mm3,
                "d_long_mm": axes["d_long_mm"],
                "d_short_mm": axes["d_short_mm"],
                "d_mean_mm": axes["d_mean_mm"],
            },
            morphology={
                "type": density_prof["nodule_type"],
                "solid_core_diameter_mm": density_prof["solid_core_diameter_mm"],
                "is_spiculated": is_spic,
            },
            risk_assessment=triad_report,
        )

        # Print Clinical Summary
        print(f"\n================================================================================")
        print(f"CLINICAL RISK ASSESSMENT REPORT -- LESION {nodule_i} ({uid[:16]}...)")
        print(f"================================================================================")
        print(f"  • Physical Volume:         {vol_mm3:.1f} mm³")
        print(f"  • Measured Diameters:      d_long={axes['d_long_mm']:.1f} mm, d_short={axes['d_short_mm']:.1f} mm, d_mean={axes['d_mean_mm']:.1f} mm")
        print(f"  • Morphological Shape:     Sphericity={sphericity:.3f} | Spiculation Index={spiculation_idx:.4f} ({'Spiculated' if is_spic else 'Smooth'})")
        print(f"  • Attenuation Subtype:     {density_prof['nodule_type'].upper()} ({density_prof['solid_core_ratio']*100:.1f}% solid core, {density_prof['ggo_ratio']*100:.1f}% GGO)")
        print(f"--------------------------------------------------------------------------------")
        print(f"  1. Calibrated ML Risk:     {triad_report['risk_percent']:.1f}% Malignancy Probability (Trained on LIDC-IDRI)")
        print(f"  2. Brock / PanCan Score:   {triad_report['brock_percent']:.1f}% Reference Probability")
        print(f"     Guideline Action:       {triad_report['bts_recommendation']}")
        print(f"  3. ACR Lung-RADS v2022:    Category {triad_report['lung_rads_category']}")
        print(f"     Clinical Management:    {triad_report['lung_rads_management']}")
        print(f"--------------------------------------------------------------------------------")
        print(f"  Concordance Status:        {triad_report['confidence_flag']}")
        print(f"================================================================================\n")

        print("[*] STANDARDIZED DIAGNOSTIC JSON PAYLOAD (Step 6 Specification):")
        print(json.dumps(diag_json, indent=2))
        print()

        # Save diagnostic visualization
        if save_viz and nodule_i == 1:
            safe_uid = uid.replace(".", "_")[:16]
            out_img_path = os.path.join(BASE_DIR, "models", f"real_scan_inference_{safe_uid}.png")
            fig, axes_pl = plt.subplots(1, 3, figsize=(15, 5))

            # Center slice of raw scan
            raw_slice = vol_hu[vox_z]
            axes_pl[0].imshow(np.clip(raw_slice, -1000, 400), cmap="gray", origin="lower")
            axes_pl[0].plot(vox_x, vox_y, "r+", markersize=14, markeredgewidth=2)
            axes_pl[0].set_title(f"Full Axial CT Slice (z={vox_z})\nRed cross: Nodule Center")
            axes_pl[0].axis("off")

            # Cropped 3D patch axial view
            patch_cz = patch_hu.shape[0] // 2
            axes_pl[1].imshow(np.clip(patch_hu[patch_cz], -1000, 400), cmap="bone", origin="lower")
            axes_pl[1].set_title(f"Resampled 3D Patch (48x48 mm)\nWL: -300, WW: 1400")
            axes_pl[1].axis("off")

            # Segmentation & Density Overlay
            overlay = np.zeros((*patch_hu[patch_cz].shape, 3), dtype=np.float32)
            norm_patch = np.clip((patch_hu[patch_cz] + 1000) / 1400.0, 0, 1)
            for c in range(3):
                overlay[..., c] = norm_patch

            mask_sl = nodule_mask_3d[patch_cz]
            overlay[mask_sl == 1, 0] = np.clip(overlay[mask_sl == 1, 0] + 0.5, 0, 1)  # Red tint for nodule
            axes_pl[2].imshow(overlay, origin="lower")
            axes_pl[2].set_title(f"3D Nodule Boundary Overlay\nType: {density_prof['nodule_type'].upper()}")
            axes_pl[2].axis("off")

            plt.suptitle(
                f"Clinical AI Inference: Scan {uid[:18]}... | ML Risk: {triad_report['risk_percent']:.1f}% | Lung-RADS {triad_report['lung_rads_category']}",
                fontsize=13,
                fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[OK] Diagnostic visualization saved to: {out_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on real LUNA16 scan")
    parser.add_argument("--scan_idx", type=int, default=0, help="Index of scan in subset0 to test")
    parser.add_argument("--series_uid", type=str, default=None, help="Specific SeriesUID to test")
    args = parser.parse_args()

    run_real_scan_test(scan_idx=args.scan_idx, series_uid=args.series_uid)
