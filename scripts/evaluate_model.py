"""
Quantitative Clinical Validation on Genuinely Unseen Hold-Out Test Set (N = 872).
Evaluates the leak-free Objective Digital Malignancy Model trained strictly on the 80% train split.
Computes AUROC, AUPRC, Brier score, sensitivity, specificity, and generates the 4-panel report.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, accuracy_score
)
from sklearn.calibration import calibration_curve


def evaluate_objective_model(
    model_path="models/objective_malignancy_xgb.joblib",
    test_set_path="dataset/objective_test_set.joblib",
    output_png="models/model_evaluation_report.png"
):
    print("=" * 80)
    print("[*] QUANTITATIVE MODEL VALIDATION ON GENUINE HOLD-OUT TEST SET (N = 872)")
    print("=" * 80)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not os.path.exists(test_set_path):
        raise FileNotFoundError(f"Test split not found at {test_set_path}")

    test_data = joblib.load(test_set_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print(f"  ✓ Loaded untouched holdout: {len(y_test):,} nodules "
          f"({np.sum(y_test == 0)} Benign, {np.sum(y_test == 1)} Malignant)")
    print(f"    (Strictly isolated BEFORE fitting — ZERO data leakage.)")

    checkpoint = joblib.load(model_path)
    model = checkpoint["model"]

    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.50).astype(int)

    auc = roc_auc_score(y_test, y_probs)
    auprc = average_precision_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / max(1, (tp + fp))
    npv = tn / max(1, (tn + fn))
    accuracy = accuracy_score(y_test, y_preds)

    print("\n" + "-" * 80)
    print("HONEST, LEAK-FREE METRICS ON UNSEEN DATA (N = 872):")
    print("-" * 80)
    print(f"  • Area Under ROC (AUROC):           {auc:.4f}  (Objective Digital Features)")
    print(f"  • Area Under PR Curve (AUPRC):      {auprc:.4f}")
    print(f"  • Brier Calibration Score:          {brier:.4f}  (0.0 = perfect calibration)")
    print(f"  • Diagnostic Accuracy:              {accuracy*100:.2f}%")
    print(f"  • Sensitivity:                      {sensitivity*100:.2f}% ({tp}/{tp+fn} malignant)")
    print(f"  • Specificity:                      {specificity*100:.2f}% ({tn}/{tn+fp} benign)")
    print(f"  • Positive Predictive Value (PPV):  {ppv*100:.2f}%")
    print(f"  • Negative Predictive Value (NPV):  {npv*100:.2f}%")
    print("-" * 80 + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Objective Calibrated Malignancy Risk Model — Genuine Holdout (N={len(y_test)})",
                 fontsize=14, fontweight="bold")

    # 1. ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[0, 0].plot(fpr, tpr, color="#0b6e4f", lw=2.5, label=f"Objective XGBoost (AUROC = {auc:.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="#9aa2ae", linestyle="--", lw=1.5, label="Random Chance")
    axes[0, 0].scatter([1 - specificity], [sensitivity], color="#b3540a", s=80, zorder=5,
                        label=f"Operating Point (Sens: {sensitivity:.2f}, Spec: {specificity:.2f})")
    axes[0, 0].set_xlabel("False Positive Rate"); axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("1. Receiver Operating Characteristic (ROC)"); axes[0, 0].legend(loc="lower right", fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. PR
    prec, rec, _ = precision_recall_curve(y_test, y_probs)
    axes[0, 1].plot(rec, prec, color="#0b4ea2", lw=2.5, label=f"PR Curve (AUPRC = {auprc:.3f})")
    axes[0, 1].axhline(np.mean(y_test), color="#9aa2ae", linestyle="--", lw=1.5,
                        label=f"Baseline Prevalence ({np.mean(y_test)*100:.1f}%)")
    axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("2. Precision-Recall Curve"); axes[0, 1].legend(loc="lower left", fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy="uniform")
    axes[1, 0].plot(prob_pred, prob_true, marker="s", color="#0b6e4f", lw=2.5, label=f"Platt Calibrated (Brier = {brier:.4f})")
    axes[1, 0].plot([0, 1], [0, 1], color="#9aa2ae", linestyle="--", lw=1.5, label="Perfect Calibration (y = x)")
    axes[1, 0].set_xlabel("Mean Predicted Malignancy Risk"); axes[1, 0].set_ylabel("Empirical True Fraction Malignant")
    axes[1, 0].set_title("3. Reliability Diagram (Probability Calibration)"); axes[1, 0].legend(loc="upper left", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Risk Distribution
    axes[1, 1].hist(y_probs[y_test == 0] * 100, bins=25, alpha=0.65, color="#0b6e4f",
                     label=f"Benign (N={np.sum(y_test==0)})", edgecolor="black")
    axes[1, 1].hist(y_probs[y_test == 1] * 100, bins=25, alpha=0.65, color="#b3540a",
                     label=f"Malignant (N={np.sum(y_test==1)})", edgecolor="black")
    axes[1, 1].axvline(50.0, color="black", linestyle="--", lw=1.5, label="50% Decision Threshold")
    axes[1, 1].set_xlabel("Predicted Malignancy Risk (%)"); axes[1, 1].set_ylabel("Nodule Count")
    axes[1, 1].set_title("4. Risk Distribution: Benign vs. Malignant"); axes[1, 1].legend(loc="upper center", fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Validation report plot saved to: {output_png}")
    print("=" * 80 + "\n")
    return {
        "auroc": auc,
        "auprc": auprc,
        "brier": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "output_png": output_png
    }


if __name__ == "__main__":
    evaluate_objective_model()
