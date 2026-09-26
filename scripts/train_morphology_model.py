"""
NEW SCRIPT — trains a malignancy classifier on ONLY features your pipeline
can genuinely compute from a real segmented nodule (volume, sphericity,
spiculation index, solid/GGO ratio, diameters). No LIDC semantic ratings,
no fabricated constants. Same leakage-safe train/holdout split as the fixed
extract_lidc_annotations.py.
"""

import os
import sys
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import xgboost as xgb

FEATURE_COLS = ["volume_mm3", "d_mean_mm", "d_long_mm", "sphericity",
                 "spiculation_index", "solid_ratio", "ggo_ratio"]


def _load_xy(csv_path):
    features, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["option_b_label"] == "":
                continue
            vals = [float(row[c]) for c in FEATURE_COLS]
            features.append(vals)
            labels.append(int(row["option_b_label"]))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def train(csv_path="dataset/lidc_morphology_features.csv",
          model_output="models/calibrated_malignancy_morphology.joblib"):
    X, y = _load_xy(csv_path)
    print(f"[*] Loaded {len(y)} real-feature nodules "
          f"({np.sum(y==0)} benign, {np.sum(y==1)} malignant)")

    if len(y) < 50:
        print("[!] WARNING: very small dataset (subset0 only covers a fraction of "
              "LIDC-IDRI). Metrics below will be noisy. Download more LUNA16 "
              "subsets and re-run build_morphology_training_set.py to grow this.")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    skf = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y_train))), shuffle=True, random_state=42)
    aucs, briers = [], []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        base = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42)
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        probs = clf.predict_proba(X_train[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train[val_idx], probs))
        briers.append(brier_score_loss(y_train[val_idx], probs))

    print(f"[OK] CV on train split — AUROC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}, "
          f"Brier: {np.mean(briers):.4f}")

    final_base = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="logloss", random_state=42)
    final_clf = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=3)
    final_clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump({"model": final_clf, "feature_cols": FEATURE_COLS}, model_output)

    holdout_probs = final_clf.predict_proba(X_holdout)[:, 1]
    print(f"[OK] Holdout AUROC: {roc_auc_score(y_holdout, holdout_probs):.3f}, "
          f"Brier: {brier_score_loss(y_holdout, holdout_probs):.4f}")
    print(f"[OK] Saved to {model_output}")


if __name__ == "__main__":
    train()
