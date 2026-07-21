from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from automl.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)


def plot_shap_summary(
    result: EvaluationResult,
    X_train_sample: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    max_background_samples: int = 100,
) -> str | None:
    """Generate a SHAP summary plot. Returns None (instead of raising) if the model
    type isn't supported by SHAP's fast explainers, since SHAP support varies by model.
    """
    model = result.model

    # Cap background sample size — SHAP can be very slow on large datasets otherwise.
    background = X_train_sample
    if background.shape[0] > max_background_samples:
        idx = np.random.RandomState(42).choice(
            background.shape[0], max_background_samples, replace=False
        )
        background = background[idx]

    try:
        if hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, GradientBoosting) — fast, exact explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background)
            # TreeExplainer on binary classifiers can return a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Linear/other models — slower, sampling-based explainer
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(background, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
    except Exception as e:
        logger.warning("SHAP explanation skipped (%s: %s)", type(e).__name__, e)
        return None

    # Guard against feature name / column count mismatch, same as feature importance plot
    if shap_values.shape[1] != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, background, feature_names=feature_names, show=False)
    fig.tight_layout()

    path = Path(output_dir) / "shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)
