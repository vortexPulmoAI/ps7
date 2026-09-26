"""
Rigorous Model Evaluation Script on Real LIDC-IDRI Hold-Out Test Split.
Evaluates the trained, calibrated XGBoost malignancy model on an independent 20% test set:
- Computes AUROC, AUPRC, Brier score, Sensitivity, Specificity, PPV, NPV
- Evaluates probability calibration via reliability curve
- Generates a 4-panel clinical validation figure saved to models/model_evaluation_report.png
"""

import os
import sys
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve


def evaluate_malignancy_model(csv_path="dataset/lidc_nodules_annotations.csv", model_path="models/calibrated_malignancy_xgb.joblib"):
    print("=" * 80)
    print("[*] QUANTITATIVE MODEL VALIDATION ON REAL LIDC-IDRI DATA")
    print("=" * 80)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotations CSV not found at {csv_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load dataset
    feature_cols = [
        "subtlety", "internal_structure", "calcification",
        "sphericity", "margin", "lobulation", "spiculation", "texture", "roi_count"
    ]
    features, labels, series_uids = [], [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl_str = row["option_b_label"]
            if lbl_str == "":
                continue
            lbl = int(lbl_str)

            vals = []
            valid = True
            for c in feature_cols:
                val = row[c]
                if val == "" or val is None:
                    valid = False
                    break
                vals.append(float(val))

            if valid:
                features.append(vals)
                labels.append(lbl)
                series_uids.append(row["series_uid"])

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"  ✓ Total Evaluated LIDC Nodules (Option B): {len(y):,}")
    print(f"    - Benign Cases    (Label 0): {np.sum(y == 0):,} ({np.mean(y == 0)*100:.1f}%)")
    print(f"    - Malignant Cases (Label 1): {np.sum(y == 1):,} ({np.mean(y == 1)*100:.1f}%)")

    # 80/20 Stratified Hold-out Split
    X_train, X_test, y_test_true, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  ✓ Independent Hold-Out Test Set: {len(y_test):,} nodules ({np.sum(y_test==0)} Benign, {np.sum(y_test==1)} Malignant)")

    # Load trained model
    saved_data = joblib.load(model_path)
    model = saved_data["model"]

    # Predict calibrated probabilities on test set
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.50).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test, y_probs)
    auprc = average_precision_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    accuracy = (tp + tn) / len(y_test)

    print("\n" + "-" * 80)
    print("HELD-OUT TEST SET METRICS (Unseen Data):")
    print("-" * 80)
    print(f"  • Area Under ROC (AUROC):           {auc:.4f}  (Target: 0.90 - 0.94)")
    print(f"  • Area Under PR Curve (AUPRC):      {auprc:.4f}")
    print(f"  • Brier Calibration Score:          {brier:.4f}  (0.0 = perfect calibration)")
    print(f"  • Diagnostic Accuracy:              {accuracy*100:.2f}%")
    print(f"  • Sensitivity (True Positive Rate): {sensitivity*100:.2f}% ({tp}/{tp+fn} malignant correctly flagged)")
    print(f"  • Specificity (True Negative Rate): {specificity*100:.2f}% ({tn}/{tn+fp} benign correctly identified)")
    print(f"  • Positive Predictive Value (PPV):  {ppv*100:.2f}%")
    print(f"  • Negative Predictive Value (NPV):  {npv*100:.2f}%")
    print("-" * 80)

    # Generate 4-Panel Clinical Validation Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Calibrated Malignancy Risk Model Validation (LIDC-IDRI Hold-Out Set, N={len(y_test)})", fontsize=14, fontweight="bold")

    # Panel 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[0, 0].plot(fpr, tpr, color="#0b6e4f", lw=2.5, label=f"Calibrated XGBoost (AUROC = {auc:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="#9aa2ae", linestyle="--", lw=1.5, label="Random Chance (0.50)")
    axes[0, 0].scatter([1 - specificity], [sensitivity], color="#b3540a", s=80, zorder=5, label=f"Operating Point (Sens: {sensitivity:.2f}, Spec: {specificity:.2f})")
    axes[0, 0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0, 0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0, 0].set_title("1. Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="semibold")
    axes[0, 0].legend(loc="lower right", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_probs)
    axes[0, 1].plot(rec, prec, color="#0b4ea2", lw=2.5, label=f"PR Curve (AUPRC = {auprc:.3f})")
    axes[0, 1].axhline(np.mean(y_test), color="#9aa2ae", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({np.mean(y_test)*100:.1f}%)")
    axes[0, 1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[0, 1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[0, 1].set_title("2. Precision-Recall Curve", fontsize=12, fontweight="semibold")
    axes[0, 1].legend(loc="lower left", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy="uniform")
    axes[1, 0].plot(prob_pred, prob_true, marker="s", color="#0b6e4f", lw=2.5, label=f"Platt Calibrated (Brier = {brier:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], color="#9aa2ae", linestyle="--", lw=1.5, label="Perfect Calibration (y = x)")
    axes[1, 0].set_xlabel("Mean Predicted Malignancy Risk", fontsize=11)
    axes[1, 0].set_ylabel("Empirical True Fraction Malignant", fontsize=11)
    axes[1, 0].set_title("3. Reliability Diagram (Probability Calibration)", fontsize=12, fontweight="semibold")
    axes[1, 0].legend(loc="upper left", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Risk Distribution Histogram
    axes[1, 1].hist(y_probs[y_test == 0] * 100, bins=25, alpha=0.6, color="#0b6e4f", label=f"Benign (N={np.sum(y_test==0)})", edgecolor="black")
    axes[1, 1].hist(y_probs[y_test == 1] * 100, bins=25, alpha=0.6, color="#b3540a", label=f"Malignant (N={np.sum(y_test==1)})", edgecolor="black")
    axes[1, 1].axvline(50.0, color="black", linestyle="--", lw=1.5, label="50% Threshold")
    axes[1, 1].set_xlabel("Predicted Malignancy Risk (%)", fontsize=11)
    axes[1, 1].set_ylabel("Nodule Count", fontsize=11)
    axes[1, 1].set_title("4. Risk Distribution: Benign vs. Malignant", fontsize=12, fontweight="semibold")
    axes[1, 1].legend(loc="upper center", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    report_img = os.path.join("models", "model_evaluation_report.png")
    plt.savefig(report_img, dpi=200)
    plt.close()

    print(f"\n[OK] Validation plot saved to: {report_img}")
    print("=" * 80 + "\n")
    return {
        "auroc": auc,
        "auprc": auprc,
        "brier": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "report_img": report_img,
    }


if __name__ == "__main__":
    evaluate_malignancy_model()
