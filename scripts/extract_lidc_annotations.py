"""
FIXED: Extract LIDC-IDRI nodule annotations, train/calibrate XGBoost malignancy model.

Fix applied: the final model is now fit ONLY on the train split. The holdout
split is saved to disk so evaluate_model.py scores on genuinely unseen data
instead of re-splitting the same CSV the full-fit model already saw.
"""

import os
import sys
import glob
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
from src.dataset_parser import LIDCXMLParser
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import xgboost as xgb

FEATURE_COLS = [
    "subtlety", "internal_structure", "calcification",
    "sphericity", "margin", "lobulation", "spiculation", "texture", "roi_count"
]


def extract_lidc_nodules_dataset(xml_root="dataset/LIDC-XML-only", output_csv="dataset/lidc_nodules_annotations.csv"):
    print(f"[*] Scanning LIDC XML files in {xml_root}...")
    xml_files = glob.glob(os.path.join(xml_root, "**", "*.xml"), recursive=True)
    print(f"    Found {len(xml_files)} XML files.")

    parser = LIDCXMLParser()
    all_records = []

    for i, xml_path in enumerate(xml_files):
        try:
            res = parser.parse_file(xml_path)
            series_uid = res["series_uid"]
            for nod in res["nodules"]:
                chars = nod["characteristics"]
                if not chars:
                    continue

                rois = nod["rois"]
                num_rois = len(rois)
                z_positions = [r["z_position"] for r in rois if r["z_position"] is not None]
                z_span = (max(z_positions) - min(z_positions)) if len(z_positions) > 1 else 0.0

                mal_score = chars.get("malignancy")
                opt_b = nod["option_b_label"]

                all_records.append({
                    "series_uid": series_uid,
                    "nodule_id": nod["nodule_id"],
                    "subtlety": chars.get("subtlety", np.nan),
                    "internal_structure": chars.get("internalStructure", np.nan),
                    "calcification": chars.get("calcification", np.nan),
                    "sphericity": chars.get("sphericity", np.nan),
                    "margin": chars.get("margin", np.nan),
                    "lobulation": chars.get("lobulation", np.nan),
                    "spiculation": chars.get("spiculation", np.nan),
                    "texture": chars.get("texture", np.nan),
                    "malignancy_score": mal_score,
                    "option_b_label": opt_b if opt_b is not None else "",
                    "roi_count": num_rois,
                    "z_span_mm": round(z_span, 2),
                })
        except Exception:
            continue

        if (i + 1) % 250 == 0 or (i + 1) == len(xml_files):
            print(f"    Processed {i + 1}/{len(xml_files)} XML files... (collected {len(all_records)} nodules)")

    fieldnames = [
        "series_uid", "nodule_id", "subtlety", "internal_structure", "calcification",
        "sphericity", "margin", "lobulation", "spiculation", "texture",
        "malignancy_score", "option_b_label", "roi_count", "z_span_mm"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"[OK] Extracted {len(all_records)} total nodule reads to {output_csv}.")
    return all_records


def _load_xy(csv_path):
    features, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl_str = row["option_b_label"]
            if lbl_str == "":
                continue
            lbl = int(lbl_str)
            vals, valid = [], True
            for col in FEATURE_COLS:
                val = row[col]
                if val == "" or val is None:
                    valid = False
                    break
                vals.append(float(val))
            if valid:
                features.append(vals)
                labels.append(lbl)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def train_calibrated_malignancy_model(csv_path="dataset/lidc_nodules_annotations.csv",
                                       model_output="models/calibrated_malignancy_xgb.joblib"):
    print("\n[*] Training Real Calibrated Malignancy Risk Model...")
    X, y = _load_xy(csv_path)

    num_benign = int(np.sum(y == 0))
    num_malignant = int(np.sum(y == 1))
    print(f"    Option B Dataset: {len(y)} total labeled nodules")
    print(f"    - Benign (labels 1-2):    {num_benign} ({num_benign / len(y) * 100:.1f}%)")
    print(f"    - Malignant (labels 4-5): {num_malignant} ({num_malignant / len(y) * 100:.1f}%)")
    print(f"    - (Indeterminate score 3 cases were dropped per benchmark standards)")

    # --- FIX: carve out the true holdout FIRST, before any fitting happens ---
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"    Train split:   {len(y_train)} nodules")
    print(f"    Holdout split: {len(y_holdout)} nodules (never touched during fit/CV)")

    # 5-fold CV on the TRAIN split only, for a model-selection estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, auprcs, briers = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xt, yt = X_train[tr_idx], y_train[tr_idx]
        Xv, yv = X_train[val_idx], y_train[val_idx]

        base_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
        )
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        calibrated_clf.fit(Xt, yt)

        val_probs = calibrated_clf.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, val_probs))
        auprcs.append(average_precision_score(yv, val_probs))
        briers.append(brier_score_loss(yv, val_probs))

    mean_auc, mean_auprc, mean_brier = float(np.mean(aucs)), float(np.mean(auprcs)), float(np.mean(briers))
    print(f"\n[OK] 5-Fold CV on TRAIN split only:")
    print(f"    - AUROC:       {mean_auc:.4f} +/- {np.std(aucs):.4f}")
    print(f"    - AUPRC:       {mean_auprc:.4f} +/- {np.std(auprcs):.4f}")
    print(f"    - Brier Score: {mean_brier:.4f} +/- {np.std(briers):.4f}")

    # --- FIX: fit final model on TRAIN split only, never on holdout ---
    final_base = xgb.XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
    )
    final_calibrated = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=5)
    final_calibrated.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump({
        "model": final_calibrated,
        "feature_cols": FEATURE_COLS,
        "metrics": {"mean_auc": mean_auc, "mean_auprc": mean_auprc, "mean_brier": mean_brier,
                    "num_train_samples": len(y_train)},
    }, model_output)
    print(f"[OK] Saved calibrated model (train-only fit) to {model_output}")

    # --- FIX: save the exact holdout so evaluate_model.py scores on truly unseen data ---
    holdout_path = os.path.join(os.path.dirname(model_output), "holdout_split.npz")
    np.savez(holdout_path, X_holdout=X_holdout, y_holdout=y_holdout)
    print(f"[OK] Saved untouched holdout split to {holdout_path}")

    return final_calibrated


if __name__ == "__main__":
    csv_file = "dataset/lidc_nodules_annotations.csv"
    if not os.path.exists(csv_file):
        extract_lidc_nodules_dataset(output_csv=csv_file)
    else:
        print(f"[*] Found existing annotations CSV at {csv_file}, proceeding to training.")
    train_calibrated_malignancy_model(csv_path=csv_file)
