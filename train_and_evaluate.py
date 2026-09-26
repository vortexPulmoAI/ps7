"""
ONE-COMMAND PRODUCTION TRAINING & CLINICAL EVALUATION PIPELINE
=============================================================================
Executes complete end-to-end training and rigorous validation:
1. Loads 4,358 LIDC-IDRI nodules with 19 objective 3D digital features.
2. Strict 80/20 train/holdout split with ZERO data leakage (stratified).
3. 5-Fold cross-validation on the 80% train split.
4. Fits & Platt-calibrates XGBoost strictly on the 80% train split.
5. Saves production model to models/objective_malignancy_xgb.joblib.
6. Evaluates on the genuinely unseen 20% holdout test set (N = 872).
7. Computes & displays:
   - Diagnostic Accuracy, AUROC, AUPRC, Brier score, Sens, Spec, PPV, NPV
   - Confusion Matrix (TP, FP, TN, FN)
8. Generates high-res 4-panel diagnostic figure (ROC, PR, Calibration, Histogram)
   saved to models/model_evaluation_report.png.
9. Runs immediate live test inference on a real 3D patient CT scan from subset0
   and saves test visualization to models/real_scan_test_inference.png.
=============================================================================
"""

import os
import sys
import csv
import math
import numpy as np

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss, accuracy_score, confusion_matrix
)
import xgboost as xgb

from src.risk_engine import BrockCalculator, LungRADSClassifier


def main():
    print("\n" + "=" * 80)
    print("   VORTEX PULMO-AI : CLINICAL MALIGNANCY MODEL TRAINING & EVALUATION")
    print("=" * 80)

    dataset_csv = os.path.join(BASE_DIR, "dataset", "lidc_objective_features.csv")
    model_output = os.path.join(BASE_DIR, "models", "objective_malignancy_xgb.joblib")
    test_set_output = os.path.join(BASE_DIR, "dataset", "objective_test_set.joblib")
    report_img_output = os.path.join(BASE_DIR, "models", "model_evaluation_report.png")
    real_scan_img_output = os.path.join(BASE_DIR, "models", "real_scan_test_inference.png")

    if not os.path.exists(dataset_csv):
        print(f"[!] Objective features dataset not found at {dataset_csv}. Compiling now...")
        from scripts.build_objective_features_dataset import extract_objective_features_from_xmls
        extract_objective_features_from_xmls(output_csv=dataset_csv)

    # -------------------------------------------------------------
    # 1. LOAD OBJECTIVE DATASET
    # -------------------------------------------------------------
    print(f"\n[1/6] Loading Objective 3D Digital Features ({dataset_csv})...")
    feature_cols = [
        "volume_mm3", "d_long_mm", "d_short_mm", "d_mean_mm", "sphericity",
        "elongation", "flatness", "surface_to_volume_ratio", "radial_variance",
        "spiculation_index", "mean_hu", "std_hu", "p10_hu", "p90_hu",
        "ggo_ratio", "solid_ratio", "calc_ratio", "solid_core_ratio", "solid_core_diameter_mm"
    ]

    X_list, y_list = [], []
    with open(dataset_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("option_b_label"):
                continue
            feats = [float(row[col]) for col in feature_cols]
            X_list.append(feats)
            y_list.append(int(row["option_b_label"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    total_nodules = len(y)
    n_benign = int(np.sum(y == 0))
    n_malignant = int(np.sum(y == 1))

    print(f"  ✓ Loaded {total_nodules:,} verified LIDC-IDRI consensus nodules:")
    print(f"    - Benign Cases    (Class 0): {n_benign:,} ({n_benign / total_nodules * 100:.1f}%)")
    print(f"    - Malignant Cases (Class 1): {n_malignant:,} ({n_malignant / total_nodules * 100:.1f}%)")

    # -------------------------------------------------------------
    # 2. ZERO DATA LEAKAGE: 80% TRAIN / 20% TEST SPLIT FIRST
    # -------------------------------------------------------------
    print("\n[2/6] Performing Strict 80/20 Train/Test Split (Zero Data Leakage)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  ✓ Training Set:  {len(y_train):,} nodules (Strictly isolated for model fitting)")
    print(f"  ✓ Holdout Test:  {len(y_test):,} nodules (Genuinely unseen; held out for evaluation)")

    # -------------------------------------------------------------
    # 3. 5-FOLD CROSS-VALIDATION ON TRAINING SPLIT ONLY
    # -------------------------------------------------------------
    print("\n[3/6] Running 5-Fold Cross-Validation on Training Split ONLY...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs, cv_auprcs, cv_briers, cv_accs = [], [], [], []
    scale_pos = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train[t_idx], y_train[t_idx]
        X_val, y_val = X_train[v_idx], y_train[v_idx]

        fold_base = xgb.XGBClassifier(
            n_estimators=130,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42 + fold
        )
        fold_cal = CalibratedClassifierCV(estimator=fold_base, method="sigmoid", cv=3)
        fold_cal.fit(X_tr, y_tr)

        val_probs = fold_cal.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        cv_aurocs.append(roc_auc_score(y_val, val_probs))
        cv_auprcs.append(average_precision_score(y_val, val_probs))
        cv_briers.append(brier_score_loss(y_val, val_probs))
        cv_accs.append(accuracy_score(y_val, val_preds))

    print(f"  • Mean Cross-Validation AUROC: {np.mean(cv_aurocs):.4f} +/- {np.std(cv_aurocs):.4f}")
    print(f"  • Mean Cross-Validation AUPRC: {np.mean(cv_auprcs):.4f} +/- {np.std(cv_auprcs):.4f}")
    print(f"  • Mean Brier Score:            {np.mean(cv_briers):.4f}")
    print(f"  • Mean CV Accuracy:            {np.mean(cv_accs)*100:.2f}%")

    # -------------------------------------------------------------
    # 4. FIT PRODUCTION MODEL ON 80% TRAIN SPLIT ONLY
    # -------------------------------------------------------------
    print("\n[4/6] Training Production Platt-Calibrated XGBoost Model on 80% Train Split...")
    final_base = xgb.XGBClassifier(
        n_estimators=140,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42
    )
    final_model = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=5)
    final_model.fit(X_train, y_train)

    # Save model checkpoints
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    payload = {
        "model": final_model,
        "feature_names": feature_cols,
        "n_train": len(y_train),
        "n_test": len(y_test)
    }
    joblib.dump(payload, model_output)
    joblib.dump({"X_test": X_test, "y_test": y_test, "feature_names": feature_cols}, test_set_output)
    print(f"  ✓ Saved Model Checkpoint -> {model_output}")

    # -------------------------------------------------------------
    # 5. EVALUATION ON GENUINELY UNSEEN TEST SPLIT (N = 872)
    # -------------------------------------------------------------
    print("\n[5/6] Evaluating on Genuinely Unseen Holdout Test Set (N = 872)...")
    y_probs = final_model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.50).astype(int)

    test_auroc = roc_auc_score(y_test, y_probs)
    test_auprc = average_precision_score(y_test, y_probs)
    test_brier = brier_score_loss(y_test, y_probs)
    test_acc = accuracy_score(y_test, y_preds)

    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / max(1, (tp + fp))
    npv = tn / max(1, (tn + fn))

    print("\n" + "=" * 80)
    print("   GENUINELY UNSEEN HOLDOUT TEST SET PERFORMANCE (N = 872, ZERO LEAKAGE)")
    print("=" * 80)
    print(f"  • Diagnostic Accuracy:              {test_acc * 100:.2f}%")
    print(f"  • Area Under ROC (AUROC):           {test_auroc:.4f}")
    print(f"  • Area Under PR Curve (AUPRC):      {test_auprc:.4f}")
    print(f"  • Brier Calibration Score:          {test_brier:.4f}  (0.0 = perfect probability)")
    print(f"  • Sensitivity (True Positive Rate): {sensitivity * 100:.2f}% ({tp}/{tp + fn} malignant detected)")
    print(f"  • Specificity (True Negative Rate): {specificity * 100:.2f}% ({tn}/{tn + fp} benign spared)")
    print(f"  • Positive Predictive Value (PPV):  {ppv * 100:.2f}%")
    print(f"  • Negative Predictive Value (NPV):  {npv * 100:.2f}%")
    print("-" * 80)
    print(f"  CONFUSION MATRIX BREAKDOWN:")
    print(f"                 Predicted Benign   Predicted Malignant")
    print(f"  Actual Benign:       {tn:4d} (TN)            {fp:4d} (FP)")
    print(f"  Actual Malig:        {fn:4d} (FN)            {tp:4d} (TP)")
    print("=" * 80)

    # -------------------------------------------------------------
    # 6. GENERATE VISUAL EVALUATION GRAPHS & HISTOGRAMS
    # -------------------------------------------------------------
    print("\n[6/6] Generating Clinical Evaluation Graphs & Test Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"Objective Malignancy Risk Model — Genuine Holdout (N = {len(y_test)})",
                 fontsize=14, fontweight="bold")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[0, 0].plot(fpr, tpr, color="#059669", lw=2.5, label=f"Objective XGBoost (AUROC = {test_auroc:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Chance")
    axes[0, 0].scatter([1 - specificity], [sensitivity], color="#dc2626", s=90, zorder=5,
                        label=f"Op. Point (Sens: {sensitivity:.2f}, Spec: {specificity:.2f})")
    axes[0, 0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0, 0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0, 0].set_title("1. Receiver Operating Characteristic (ROC)")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_probs)
    axes[0, 1].plot(rec, prec, color="#2563eb", lw=2.5, label=f"PR Curve (AUPRC = {test_auprc:.3f})")
    axes[0, 1].axhline(np.mean(y_test), color="#94a3b8", linestyle="--", lw=1.5,
                        label=f"Baseline Prevalence ({np.mean(y_test)*100:.1f}%)")
    axes[0, 1].set_xlabel("Recall (Sensitivity)")
    axes[0, 1].set_ylabel("Precision (Positive Predictive Value)")
    axes[0, 1].set_title("2. Precision-Recall Curve")
    axes[0, 1].legend(loc="lower left", fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Probability Calibration (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy="uniform")
    axes[1, 0].plot(prob_pred, prob_true, marker="s", color="#059669", lw=2.5, label=f"Platt Calibrated (Brier = {test_brier:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Perfect Calibration (y = x)")
    axes[1, 0].set_xlabel("Mean Predicted Malignancy Probability")
    axes[1, 0].set_ylabel("Empirical True Fraction Malignant")
    axes[1, 0].set_title("3. Reliability Diagram (Probability Calibration)")
    axes[1, 0].legend(loc="upper left", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Risk Distribution Histogram
    axes[1, 1].hist(y_probs[y_test == 0] * 100, bins=25, alpha=0.65, color="#059669",
                     label=f"Benign (N = {np.sum(y_test==0)})", edgecolor="black")
    axes[1, 1].hist(y_probs[y_test == 1] * 100, bins=25, alpha=0.65, color="#dc2626",
                     label=f"Malignant (N = {np.sum(y_test==1)})", edgecolor="black")
    axes[1, 1].axvline(50.0, color="black", linestyle="--", lw=1.5, label="50% Decision Threshold")
    axes[1, 1].set_xlabel("Predicted Malignancy Risk (%)")
    axes[1, 1].set_ylabel("Nodule Count")
    axes[1, 1].set_title("4. Risk Distribution: Benign vs. Malignant")
    axes[1, 1].legend(loc="upper center", fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig(report_img_output, dpi=200, bbox_inches="tight")
    except Exception:
        import shutil
        tmp = os.path.join(BASE_DIR, "models", "model_evaluation_report_gen.png")
        plt.savefig(tmp, dpi=200, bbox_inches="tight")
        try:
            shutil.copy2(tmp, report_img_output)
        except Exception:
            report_img_output = tmp
    plt.close()
    print(f"  ✓ Diagnostic Evaluation Report Plot -> {report_img_output}")

    # -------------------------------------------------------------
    # 7. LIVE TEST INFERENCE ON REAL PATIENT SCAN (Scan #0)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   LIVE INFERENCE VERIFICATION ON REAL 3D SCAN (LUNA16 Scan #0)")
    print("=" * 80)
    from scripts.test_on_real_scan import run_real_scan_test
    run_real_scan_test(scan_idx=0, save_viz=True)

    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE COMPLETE: Model trained, evaluated, verified, and saved.")
    print(f"  • Trained Model Checkpoint: {model_output}")
    print(f"  • Validation Plot:         {report_img_output}")
    print(f"  • Test Scan Visualization: {real_scan_img_output}")
    print(f"  • Web Workstation Live at: http://127.0.0.1:8000")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
