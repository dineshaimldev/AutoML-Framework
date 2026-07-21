"""Generate evaluation plots: confusion matrix, ROC curve, feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed — we're saving files, not showing windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from automl.evaluation.metrics import EvaluationResult


def plot_confusion_matrix(result: EvaluationResult, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        result.confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted: No Churn", "Predicted: Churn"],
        yticklabels=["Actual: No Churn", "Actual: Churn"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_roc_curve(result: EvaluationResult, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result.fpr, result.tpr, label=f"ROC curve (AUC = {result.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = Path(output_dir) / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_feature_importance(
    result: EvaluationResult, feature_names: list[str], output_dir: str, top_n: int = 15
) -> str | None:
    """Plot feature importance if the model supports it (tree-based models do; SVM/logreg don't
    expose the same attribute, so we return None gracefully instead of crashing).
    """
    model = result.model

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For linear models, use absolute coefficient magnitude as a proxy for importance
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Guard against a mismatch between feature_names length and importances length
    # (can happen if one-hot encoding expanded columns) — fall back to generic names.
    if len(importances) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[order]
    top_names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, max(4, len(top_names) * 0.3)))
    ax.barh(top_names[::-1], top_importances[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top_names)} Feature Importances")
    fig.tight_layout()

    path = Path(output_dir) / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
