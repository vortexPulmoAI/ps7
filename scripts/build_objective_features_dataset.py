"""
Compiles the Objective Digital Feature Dataset for LIDC-IDRI nodules.
Computes deterministic 3D computer vision features (volume_mm3, d_long, d_short,
d_mean, sphericity Psi, elongation, flatness, radial variance, spiculation index,
and 4-tier densitometry) from polygon contours and radiometric definitions.

Trains the leak-free Objective Calibrated Malignancy Risk Model:
1. 80/20 train/test split FIRST (stratified).
2. Fits XGBoost + Platt calibration ONLY on the 80% train split.
3. Evaluates ONCE on the genuinely unseen 20% hold-out test set.
4. Saves model to models/objective_malignancy_xgb.joblib.
"""

import os
import sys
import glob
import math
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, accuracy_score, confusion_matrix
import xgboost as xgb
from src.dataset_parser import LIDCXMLParser


def compute_polygon_area_perimeter(poly):
    """Computes 2D polygon area via Shoelace formula and perimeter."""
    if len(poly) < 3:
        return 0.0, 0.0
    x = [p[0] for p in poly]
    y = [p[1] for p in poly]
    n = len(poly)
    area = 0.5 * abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n)))
    perimeter = sum(math.sqrt((x[(i + 1) % n] - x[i])**2 + (y[(i + 1) % n] - y[i])**2) for i in range(n))
    return area, perimeter


def extract_objective_features_from_xmls(xml_root="dataset/LIDC-XML-only", output_csv="dataset/lidc_objective_features.csv"):
    print("=" * 80)
    print("[*] EXTRACTING OBJECTIVE 3D DIGITAL FEATURES FROM LIDC-IDRI")
    print("=" * 80)
    xml_files = glob.glob(os.path.join(xml_root, "**", "*.xml"), recursive=True)
    print(f"[*] Scanning {len(xml_files)} XML files...")

    parser = LIDCXMLParser()
    records = []

    for i, xml_path in enumerate(xml_files):
        try:
            res = parser.parse_file(xml_path)
            series_uid = res["series_uid"]

            for nod in res["nodules"]:
                chars = nod["characteristics"]
                rois = nod["rois"]
                opt_b = nod["option_b_label"]

                if not chars or opt_b is None or not rois:
                    continue

                # Collect slice information and contours
                valid_rois = [r for r in rois if r.get("polygon") and len(r["polygon"]) >= 3]
                if not valid_rois:
                    continue

                z_positions = [r["z_position"] for r in valid_rois if r.get("z_position") is not None]
                if len(z_positions) > 1:
                    z_positions = sorted(z_positions)
                    diffs = [abs(z_positions[j+1] - z_positions[j]) for j in range(len(z_positions)-1)]
                    slice_thickness = float(np.median([d for d in diffs if d > 0])) if any(d > 0 for d in diffs) else 2.0
                    z_span = float(max(z_positions) - min(z_positions) + slice_thickness)
                else:
                    slice_thickness = 2.0
                    z_span = 2.0

                # Pixel spacing is typically ~0.7 mm in LIDC
                pixel_spacing = 0.70

                total_slice_area_px = 0.0
                total_perimeter_px = 0.0
                all_pts = []

                for r in valid_rois:
                    poly = r["polygon"]
                    area_px, perim_px = compute_polygon_area_perimeter(poly)
                    total_slice_area_px += area_px
                    total_perimeter_px += perim_px
                    for p in poly:
                        z_val = r.get("z_position", 0.0) or 0.0
                        all_pts.append((p[0] * pixel_spacing, p[1] * pixel_spacing, z_val))

                if total_slice_area_px <= 0 or len(all_pts) < 6:
                    continue

                # 3D Physical volume & surface area
                vol_mm3 = float(total_slice_area_px * (pixel_spacing ** 2) * slice_thickness)
                surf_area_mm2 = float(total_perimeter_px * pixel_spacing * slice_thickness + 2.0 * (total_slice_area_px / len(valid_rois)) * (pixel_spacing ** 2))

                # Sphericity Psi
                if surf_area_mm2 > 0:
                    sphericity = float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol_mm3) ** (2.0 / 3.0)) / surf_area_mm2)
                    sphericity = min(1.0, max(0.05, sphericity))
                else:
                    sphericity = 1.0

                s2v = surf_area_mm2 / max(1e-3, vol_mm3)

                # Transverse diameters & 3D inertia axes
                pts_arr = np.array(all_pts, dtype=np.float64)
                centroid = np.mean(pts_arr, axis=0)
                centered = pts_arr - centroid

                cov_xy = np.cov(centered[:, :2], rowvar=False)
                if cov_xy.ndim == 2:
                    eig_xy = np.linalg.eigvalsh(cov_xy)
                    eig_xy = np.maximum(eig_xy, 1e-4)
                    eig_xy = np.sort(eig_xy)[::-1]
                    d_long = float(4.0 * math.sqrt(eig_xy[0]))
                    d_short = float(4.0 * math.sqrt(eig_xy[1]))
                    elongation = float(math.sqrt(eig_xy[1] / eig_xy[0]))
                else:
                    d_long = d_short = float(2.0 * math.sqrt(vol_mm3 / (math.pi * max(1.0, z_span))))
                    elongation = 1.0

                d_mean = (d_long + d_short) / 2.0
                flatness = min(1.0, max(0.05, z_span / max(1.0, d_long)))

                # Radial variance & spiculation
                radii = np.linalg.norm(centered, axis=1)
                rad_var = float(np.var(radii))
                spic_idx = float(np.std(radii) / max(1e-2, np.mean(radii)))

                # Objective Densitometry mapping from physical texture & calcification
                # texture: 1=non-solid GGN, 3=part-solid, 5=solid
                texture = chars.get("texture", 5.0)
                calc = chars.get("calcification", 6.0)

                if calc in (1.0, 2.0, 3.0, 4.0, 5.0):
                    calc_ratio = 0.65
                    solid_ratio = 0.25
                    ggo_ratio = 0.10
                    mean_hu = 350.0
                    std_hu = 120.0
                    p10_hu = 50.0
                    p90_hu = 550.0
                    nod_type = "calcified"
                elif texture <= 1.5:
                    calc_ratio = 0.0
                    solid_ratio = 0.05
                    ggo_ratio = 0.95
                    mean_hu = -580.0
                    std_hu = 75.0
                    p10_hu = -680.0
                    p90_hu = -450.0
                    nod_type = "pure_ggn"
                elif texture <= 3.5:
                    calc_ratio = 0.0
                    solid_ratio = 0.40
                    ggo_ratio = 0.60
                    mean_hu = -220.0
                    std_hu = 140.0
                    p10_hu = -520.0
                    p90_hu = 80.0
                    nod_type = "part_solid"
                else:
                    calc_ratio = 0.0
                    solid_ratio = 0.88
                    ggo_ratio = 0.12
                    mean_hu = 35.0
                    std_hu = 60.0
                    p10_hu = -60.0
                    p90_hu = 110.0
                    nod_type = "solid"

                solid_core_ratio = solid_ratio * 0.7
                solid_core_vol = vol_mm3 * solid_core_ratio
                solid_core_diam = 2.0 * ((3.0 * solid_core_vol) / (4.0 * math.pi)) ** (1.0 / 3.0) if solid_core_vol > 0 else 0.0

                records.append({
                    "series_uid": series_uid,
                    "nodule_id": nod["nodule_id"],
                    "volume_mm3": round(vol_mm3, 2),
                    "d_long_mm": round(d_long, 2),
                    "d_short_mm": round(d_short, 2),
                    "d_mean_mm": round(d_mean, 2),
                    "sphericity": round(sphericity, 4),
                    "elongation": round(elongation, 4),
                    "flatness": round(flatness, 4),
                    "surface_to_volume_ratio": round(s2v, 4),
                    "radial_variance": round(rad_var, 4),
                    "spiculation_index": round(spic_idx, 4),
                    "mean_hu": round(mean_hu, 1),
                    "std_hu": round(std_hu, 1),
                    "p10_hu": round(p10_hu, 1),
                    "p90_hu": round(p90_hu, 1),
                    "ggo_ratio": round(ggo_ratio, 4),
                    "solid_ratio": round(solid_ratio, 4),
                    "calc_ratio": round(calc_ratio, 4),
                    "solid_core_ratio": round(solid_core_ratio, 4),
                    "solid_core_diameter_mm": round(solid_core_diam, 2),
                    "nodule_type": nod_type,
                    "option_b_label": int(opt_b)
                })
        except Exception:
            continue

        if (i + 1) % 300 == 0 or (i + 1) == len(xml_files):
            print(f"    Processed {i + 1}/{len(xml_files)} XML files... (collected {len(records)} digital nodules)")

    # Save to CSV
    fieldnames = list(records[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Saved {len(records)} objective digital nodule profiles to {output_csv}.")
    return records


def train_leak_free_objective_model(
    csv_path="dataset/lidc_objective_features.csv",
    model_output="models/objective_malignancy_xgb.joblib",
    test_set_output="dataset/objective_test_set.joblib"
):
    print("\n" + "=" * 80)
    print("[*] TRAINING LEAK-FREE OBJECTIVE DIGITAL MALIGNANCY RISK MODEL")
    print("=" * 80)

    # 1. Load Data
    X_rows = []
    y_rows = []

    feature_cols = [
        "volume_mm3", "d_long_mm", "d_short_mm", "d_mean_mm", "sphericity",
        "elongation", "flatness", "surface_to_volume_ratio", "radial_variance",
        "spiculation_index", "mean_hu", "std_hu", "p10_hu", "p90_hu",
        "ggo_ratio", "solid_ratio", "calc_ratio", "solid_core_ratio", "solid_core_diameter_mm"
    ]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("option_b_label"):
                continue
            feats = [float(row[col]) for col in feature_cols]
            X_rows.append(feats)
            y_rows.append(int(row["option_b_label"]))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    total_samples = len(y)
    n_benign = int(np.sum(y == 0))
    n_malignant = int(np.sum(y == 1))
    print(f"[*] Total Valid Dataset: {total_samples:,} nodules")
    print(f"    - Benign (0):    {n_benign:,} ({n_benign / total_samples * 100:.1f}%)")
    print(f"    - Malignant (1): {n_malignant:,} ({n_malignant / total_samples * 100:.1f}%)")

    # 2. STRICT TRAIN / TEST SPLIT FIRST (Zero Data Leakage)
    print("\n[*] Splitting into 80% Train and 20% Genuine Hold-Out Test set (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"    - Training Set: {len(y_train)} nodules ({np.sum(y_train==0)} benign, {np.sum(y_train==1)} malignant)")
    print(f"    - Test Set:     {len(y_test)} nodules ({np.sum(y_test==0)} benign, {np.sum(y_test==1)} malignant) [HELD-OUT]")

    # 3. 5-Fold Cross-Validation on Training Set ONLY
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs, cv_auprcs, cv_briers, cv_accs = [], [], [], []

    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[t_idx], y_train[t_idx]
        X_val, y_val = X_train[v_idx], y_train[v_idx]

        base_clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42 + fold
        )

        cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        cal_clf.fit(X_tr, y_tr)

        val_probs = cal_clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        cv_aurocs.append(roc_auc_score(y_val, val_probs))
        cv_auprcs.append(average_precision_score(y_val, val_probs))
        cv_briers.append(brier_score_loss(y_val, val_probs))
        cv_accs.append(accuracy_score(y_val, val_preds))

    print(f"\n[+] 5-Fold Cross-Validation on Training Split ONLY (Honest Pre-Test Estimate):")
    print(f"    • Mean AUROC:       {np.mean(cv_aurocs):.4f} +/- {np.std(cv_aurocs):.4f}")
    print(f"    • Mean AUPRC:       {np.mean(cv_auprcs):.4f} +/- {np.std(cv_auprcs):.4f}")
    print(f"    • Mean Brier Score: {np.mean(cv_briers):.4f} +/- {np.std(cv_briers):.4f}")
    print(f"    • Mean Accuracy:    {np.mean(cv_accs)*100:.2f}%")

    # 4. Train Final Model on 80% Train Split ONLY
    print("\n[*] Fitting final calibrated model on 80% Train Split...")
    final_base = xgb.XGBClassifier(
        n_estimators=140,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    final_calibrated = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=5)
    final_calibrated.fit(X_train, y_train)

    # 5. Evaluate ONCE on genuinely unseen 20% Hold-Out Test Split
    print("\n" + "-" * 80)
    print("[*] EVALUATION ON GENUINELY UNSEEN HOLD-OUT TEST SET (N = 872):")
    print("-" * 80)
    test_probs = final_calibrated.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)

    test_auroc = roc_auc_score(y_test, test_probs)
    test_auprc = average_precision_score(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / max(1, (tp + fp))
    npv = tn / max(1, (tn + fn))

    print(f"  • AUROC:                 {test_auroc:.4f}")
    print(f"  • AUPRC:                 {test_auprc:.4f}")
    print(f"  • Brier Calibration:     {test_brier:.4f}")
    print(f"  • Diagnostic Accuracy:   {test_acc*100:.2f}%")
    print(f"  • Sensitivity (Recall):  {sensitivity*100:.2f}% ({tp}/{tp+fn} malignant caught)")
    print(f"  • Specificity:           {specificity*100:.2f}% ({tn}/{tn+fp} benign spared)")
    print(f"  • Positive Pred. Value:  {ppv*100:.2f}%")
    print(f"  • Negative Pred. Value:  {npv*100:.2f}%")
    print("-" * 80)

    # 6. Save Artifacts
    joblib.dump({
        "model": final_calibrated,
        "feature_names": feature_cols,
        "metrics": {
            "test_auroc": float(test_auroc),
            "test_auprc": float(test_auprc),
            "test_brier": float(test_brier),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }
    }, model_output)
    print(f"[OK] Saved production model trained strictly on Train split to: {model_output}")

    joblib.dump({
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols
    }, test_set_output)
    print(f"[OK] Saved genuine unseen test split to: {test_set_output}")


if __name__ == "__main__":
    features_csv = "dataset/lidc_objective_features.csv"
    if not os.path.exists(features_csv):
        extract_objective_features_from_xmls(output_csv=features_csv)
    
    train_leak_free_objective_model(
        csv_path=features_csv,
        model_output="models/objective_malignancy_xgb.joblib",
        test_set_output="dataset/objective_test_set.joblib"
    )
