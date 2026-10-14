import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def calculate_evaluation_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: List[float],
) -> Dict[str, Any]:
    """
    Computes standard scikit-learn binary classification metrics.
    Class 0 = FAKE, Class 1 = REAL.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_score_arr = np.array(y_score)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion matrix with explicit [0, 1] labels
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(val) for val in cm.ravel()]

    # Specificity and error rates
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC with single-class guard
    roc_auc: Optional[float] = None
    roc_auc_status = "ok"
    if len(np.unique(y_true_arr)) < 2:
        roc_auc_status = "unavailable_single_class_present"
        logger.warning("ROC-AUC unavailable: test set contains only one unique class.")
    else:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
        except Exception as exc:
            roc_auc_status = f"error: {exc}"
            logger.error("Failed to compute ROC-AUC: %s", exc)

    clf_report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=["FAKE", "REAL"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_auc_status": roc_auc_status,
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_evaluated": len(y_true),
        "confusion_matrix_raw": [[tn, fp], [fn, tp]],
        "classification_report": clf_report,
    }


def generate_confusion_matrix_plot(cm: np.ndarray, model_name: str, output_path: Path):
    """
    Renders an annotated, publication-quality Confusion Matrix heatmap.
    """
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["FAKE (0)", "REAL (1)"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_name}",
        ylabel="Actual Ground Truth",
        xlabel="Model Prediction",
    )

    thresh = cm.max() / 2.0
    labels = [
        [f"TN: {tn}", f"FP: {fp}"],
        [f"FN: {fn}", f"TP: {tp}"],
    ]

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n({cm[i, j]})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", output_path)


def generate_roc_curve_plot(
    y_true: List[int],
    y_score: List[float],
    roc_auc: Optional[float],
    model_name: str,
    output_path: Path,
):
    """
    Plots Receiver Operating Characteristic (ROC) curve with AUC label.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    if roc_auc is not None and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
        ax.set_title(f"ROC Curve: {model_name}", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    else:
        ax.text(
            0.5,
            0.5,
            "ROC Curve Unavailable\n(Single class present in test split)",
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
        )
        ax.set_title(f"ROC Curve: {model_name}", fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve plot to %s", output_path)
