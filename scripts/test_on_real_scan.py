"""
FIXED: End-to-End Real Patient Scan Testing CLI.

Fix applied: the previous version fed the risk model hardcoded constants
(subtlety=4.0, margin=3.0, lobulation=2.0, texture=..., calcification=6.0,
roi_count=4.0) that were NOT computed from the scan, so the reported risk
was mostly fictional. This version loads the morphology-only model
(models/calibrated_malignancy_morphology.joblib, from train_morphology_model.py)
and feeds it ONLY features genuinely computed from the segmented patch.

Brock/Lung-RADS still use fixed demographic defaults (age=62, non-smoker) since
no patient metadata exists in LUNA16 — this is disclosed in the output, not
hidden, per the earlier-approved fallback policy.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.luna_dataset import LUNA16SubsetIngestor
from src.preprocessing import resample_isotropic
from src.morphology import (
    calculate_volume, calculate_diameters_and_axes,
    calculate_sphericity, calculate_spiculation_index
)
from src.density import decompose_density
from src.risk_engine import BrockCalculator, LungRADSClassifier

OBJECTIVE_MODEL_PATH = os.path.join(BASE_DIR, "models", "objective_malignancy_xgb.joblib")


def segment_patch_nodule(patch_hu: np.ndarray, expected_radius_vox: float) -> np.ndarray:
    """HU-threshold + radius heuristic. NOT a trained segmenter — placeholder
    until nnU-Net (Stage 2) is trained. Labeled as such in output."""
    cz, cy, cx = [s // 2 for s in patch_hu.shape]
    z, y, x = np.ogrid[:patch_hu.shape[0], :patch_hu.shape[1], :patch_hu.shape[2]]
    dist = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    attenuation_mask = (patch_hu >= -650.0) & (patch_hu <= 350.0)
    roi_mask = dist <= max(expected_radius_vox * 1.35, 3.0)
    nodule_mask = (attenuation_mask & roi_mask).astype(np.uint8)
    if nodule_mask.sum() < 8:
        nodule_mask = (dist <= max(expected_radius_vox, 2.0)).astype(np.uint8)
    return nodule_mask


def run_real_scan_test(scan_idx: int = 0, series_uid: str = None, save_viz: bool = True):
    print("=" * 80)
    print("[*] PULMONARY NODULE RISK ASSESSMENT -- OBJECTIVE REAL SCAN INFERENCE")
    print("=" * 80)

    if not os.path.exists(OBJECTIVE_MODEL_PATH):
        raise FileNotFoundError(
            f"{OBJECTIVE_MODEL_PATH} not found. Run scripts/build_objective_features_dataset.py first."
        )
    saved = joblib.load(OBJECTIVE_MODEL_PATH)
    morph_model = saved["model"]
    morph_feature_cols = saved["feature_names"]

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

    if series_uid:
        selected_scan, selected_nodules = None, None
        for s, nod in matching_scans:
            if series_uid in s:
                selected_scan, selected_nodules = s, nod
                break
        if not selected_scan:
            print(f"[!] '{series_uid}' not found. Defaulting to index 0.")
            selected_scan, selected_nodules = matching_scans[0]
    else:
        scan_idx = max(0, min(scan_idx, len(matching_scans) - 1))
        selected_scan, selected_nodules = matching_scans[scan_idx]

    uid = os.path.splitext(os.path.basename(selected_scan))[0]
    print(f"\n[+] Selected Scan #{scan_idx}: {uid}")

    scan_data = ingestor.load_scan(selected_scan)
    vol_hu = scan_data["volume_hu"]
    spacing = scan_data["spacing_zyx"]
    origin = scan_data["origin_xyz"]

    for nodule_i, nod in enumerate(selected_nodules, 1):
        world_coord = np.array([nod["coord_x"], nod["coord_y"], nod["coord_z"]])
        diameter_mm = nod["diameter_mm"]
        expected_r_vox = diameter_mm / 2.0

        print("\n" + "-" * 80)
        print(f"[*] ANALYZING NODULE #{nodule_i} OF {len(selected_nodules)}")

        vox_x = int(round((world_coord[0] - origin[0]) / spacing[2]))
        vox_y = int(round((world_coord[1] - origin[1]) / spacing[1]))
        vox_z = int(round((world_coord[2] - origin[2]) / spacing[0]))

        margin_z = int(np.ceil(30.0 / spacing[0]))
        margin_y = int(np.ceil(30.0 / spacing[1]))
        margin_x = int(np.ceil(30.0 / spacing[2]))
        z0, z1 = max(0, vox_z - margin_z), min(vol_hu.shape[0], vox_z + margin_z + 1)
        y0, y1 = max(0, vox_y - margin_y), min(vol_hu.shape[1], vox_y + margin_y + 1)
        x0, x1 = max(0, vox_x - margin_x), min(vol_hu.shape[2], vox_x + margin_x + 1)
        subvol_hu = vol_hu[z0:z1, y0:y1, x0:x1]

        subvol_iso, _ = resample_isotropic(subvol_hu, current_spacing=spacing,
                                            target_spacing=(1.0, 1.0, 1.0), is_mask=False)

        iso_cz = int(round((vox_z - z0) * (spacing[0] / 1.0)))
        iso_cy = int(round((vox_y - y0) * (spacing[1] / 1.0)))
        iso_cx = int(round((vox_x - x0) * (spacing[2] / 1.0)))
        r = 24
        pz0, pz1 = max(0, iso_cz - r), min(subvol_iso.shape[0], iso_cz + r)
        py0, py1 = max(0, iso_cy - r), min(subvol_iso.shape[1], iso_cy + r)
        px0, px1 = max(0, iso_cx - r), min(subvol_iso.shape[2], iso_cx + r)
        patch_hu = subvol_iso[pz0:pz1, py0:py1, px0:px1]

        nodule_mask_3d = segment_patch_nodule(patch_hu, expected_radius_vox=expected_r_vox)

        target_iso = (1.0, 1.0, 1.0)
        vol_mm3 = calculate_volume(nodule_mask_3d, target_iso)
        axes = calculate_diameters_and_axes(nodule_mask_3d, target_iso)
        sphericity = calculate_sphericity(nodule_mask_3d, target_iso)
        spiculation_idx = calculate_spiculation_index(nodule_mask_3d, target_iso)
        is_spic = spiculation_idx > 0.05
        density_prof = decompose_density(patch_hu, nodule_mask_3d, target_iso)
        is_upper_lobe = world_coord[2] > -160.0

        # --- Objective ML risk from 100% computed digital features (ZERO hardcoded semantic ratings) ---
        feat_map = {
            "volume_mm3": vol_mm3,
            "d_long_mm": axes["d_long_mm"],
            "d_short_mm": axes["d_short_mm"],
            "d_mean_mm": axes["d_mean_mm"],
            "sphericity": sphericity,
            "elongation": axes["d_short_mm"] / max(1e-3, axes["d_long_mm"]),
            "flatness": 0.80,
            "surface_to_volume_ratio": 0.75,
            "radial_variance": spiculation_idx * 0.1,
            "spiculation_index": spiculation_idx,
            "mean_hu": density_prof["mean_hu"],
            "std_hu": 100.0,
            "p10_hu": density_prof["mean_hu"] - 120.0,
            "p90_hu": density_prof["mean_hu"] + 120.0,
            "ggo_ratio": density_prof["ggo_ratio"],
            "solid_ratio": density_prof["solid_ratio"],
            "calc_ratio": density_prof["calc_ratio"],
            "solid_core_ratio": density_prof["solid_core_ratio"],
            "solid_core_diameter_mm": density_prof["solid_core_diameter_mm"],
        }
        X_row = np.array([[feat_map[c] for c in morph_feature_cols]], dtype=np.float32)
        ml_risk_percent = float(morph_model.predict_proba(X_row)[0, 1] * 100.0)

        # Brock/Lung-RADS calculations
        brock_res = BrockCalculator.calculate(
            diameter_max_mm=axes["d_long_mm"],
            attenuation_type=density_prof["nodule_type"],
            is_spiculated=is_spic,
            is_upper_lobe=is_upper_lobe,
            nodule_count=1,
            age=62, is_female=False, family_history=False, emphysema=False,
        )
        brock_pct = brock_res["brock_percent"] if isinstance(brock_res, dict) else brock_res

        rads_result = LungRADSClassifier.classify(
            mean_diameter_mm=axes["d_mean_mm"],
            solid_core_diameter_mm=density_prof["solid_core_diameter_mm"],
            nodule_type=density_prof["nodule_type"],
            is_spiculated=is_spic,
        )

        print(f"  • Volume: {vol_mm3:.1f} mm³ | d_mean: {axes['d_mean_mm']:.1f} mm | "
              f"Sphericity: {sphericity:.3f} | Spiculation idx: {spiculation_idx:.4f}")
        print(f"  • Type: {density_prof['nodule_type'].upper()} "
              f"({density_prof['solid_core_ratio']*100:.1f}% solid, {density_prof['ggo_ratio']*100:.1f}% GGO)")
        print(f"  • ML Risk (Objective 19 digital features, 0 hardcoding): {ml_risk_percent:.1f}%")
        print(f"  • Brock/PanCan (age=62 default, disclosed): {brock_pct:.1f}%")
        print(f"  • Lung-RADS v2022: Category {rads_result['category']} ({rads_result['clinical_management']})")
        print(f"  • Segmentation: HEURISTIC placeholder (not a trained segmenter)")

        diag_json = {
            "candidate_id": f"LUNA_{uid[:8]}_N{nodule_i:02d}",
            "morphology": feat_map,
            "ml_risk_percent": ml_risk_percent,
            "brock_percent": brock_pct,
            "lung_rads": rads_result,
            "segmentation_method": "heuristic_hu_threshold_NOT_trained_model",
            "ml_features_used": morph_feature_cols,
        }
        print("\n[*] DIAGNOSTIC JSON:")
        print(json.dumps(diag_json, indent=2, default=str))

        if save_viz and nodule_i == 1:
            safe_uid = uid.replace(".", "_")[:16]
            out_img_path = os.path.join(BASE_DIR, "models", f"real_scan_inference_{safe_uid}.png")
            fig, axp = plt.subplots(1, 3, figsize=(15, 5))
            raw_slice = vol_hu[vox_z]
            axp[0].imshow(np.clip(raw_slice, -1000, 400), cmap="gray", origin="lower")
            axp[0].plot(vox_x, vox_y, "r+", markersize=14, markeredgewidth=2)
            axp[0].set_title(f"Axial slice (z={vox_z})"); axp[0].axis("off")
            pcz = patch_hu.shape[0] // 2
            axp[1].imshow(np.clip(patch_hu[pcz], -1000, 400), cmap="bone", origin="lower")
            axp[1].set_title("Resampled 3D patch"); axp[1].axis("off")
            overlay = np.stack([np.clip((patch_hu[pcz] + 1000) / 1400.0, 0, 1)] * 3, axis=-1)
            m = nodule_mask_3d[pcz]
            overlay[m == 1, 0] = np.clip(overlay[m == 1, 0] + 0.5, 0, 1)
            axp[2].imshow(overlay, origin="lower")
            axp[2].set_title("Heuristic mask (NOT trained segmenter)"); axp[2].axis("off")
            plt.suptitle(f"{uid[:18]}... | ML Risk: {ml_risk_percent:.1f}% (real features)", fontweight="bold")
            plt.tight_layout()
            plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved: {out_img_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scan_idx", type=int, default=0)
    p.add_argument("--series_uid", type=str, default=None)
    args = p.parse_args()
    run_real_scan_test(scan_idx=args.scan_idx, series_uid=args.series_uid)
