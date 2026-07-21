"""MLflow integration: logs every Optuna trial and the final best model.

We wrap MLflow calls in this one module so the rest of the codebase never
imports mlflow directly — if you ever swap tracking backends (e.g. to
Weights & Biases), only this file needs to change.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import mlflow
import optuna

from automl.config import AutoMLConfig
from automl.evaluation.metrics import EvaluationResult

if TYPE_CHECKING:
    from automl.optimization.search import ModelSearchResult


def _normalize_tracking_uri(tracking_uri: str) -> str:
    """Ensure a local filesystem tracking URI is valid for MLflow.

    Plain folder paths are created and converted to file URIs so MLflow can
    use them reliably across platforms, including Windows drive-letter paths.
    URIs that already specify a scheme (sqlite:///, http://, https://, file://)
    are returned unchanged.
    """
    if "://" in tracking_uri:
        return tracking_uri

    path = Path(tracking_uri)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def setup_mlflow(config: AutoMLConfig) -> None:
    """Point MLflow at the local tracking directory and set the experiment."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = _normalize_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)


@contextmanager
def mlflow_run(run_name: str) -> Iterator[None]:
    """Context manager so callers just do `with mlflow_run("name"): ...` and
    never have to remember to call mlflow.end_run() themselves, even on error.
    """
    mlflow.start_run(run_name=run_name)
    try:
        yield
    finally:
        mlflow.end_run()


def log_trial_as_nested_run(
    trial: optuna.Trial, model_name: str, metric_name: str, score: float
) -> None:
    """Best-effort logging for Optuna trials.

    MLflow can raise under newer versions or unusual URIs; the search workflow
    should continue even if tracking is unavailable.
    """
    try:
        with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric(metric_name, score)
            mlflow.set_tag("model_name", model_name)
    except Exception:
        return


def log_model_search_summary(result: "ModelSearchResult", metric_name: str) -> None:
    """Best-effort logging for the best-trial summary for one model type."""
    try:
        mlflow.log_metric(f"{result.model_name}_best_{metric_name}", result.best_score)
        for param_name, param_value in result.best_params.items():
            mlflow.log_param(f"{result.model_name}_{param_name}", param_value)
    except Exception:
        return


def log_final_evaluation(evaluation: EvaluationResult, best_model_name: str) -> None:
    """Best-effort logging for the winning model's final metrics and artifact."""
    try:
        mlflow.set_tag("winning_model", best_model_name)
        mlflow.log_metric("test_accuracy", evaluation.accuracy)
        mlflow.log_metric("test_precision", evaluation.precision)
        mlflow.log_metric("test_recall", evaluation.recall)
        mlflow.log_metric("test_f1", evaluation.f1)
        mlflow.log_metric("test_roc_auc", evaluation.roc_auc)
        mlflow.sklearn.log_model(
            evaluation.model,
            "model",
            skops_trusted_types=[
                "sklearn.ensemble._gb_losses.BinomialDeviance",
                "sklearn.ensemble._gb_losses.ExponentialDeviance",
            ],
        )
    except Exception:
        return


def log_report_artifacts(report_dir: str) -> None:
    """Best-effort report artifact logging when MLflow is available."""
    try:
        mlflow.log_artifacts(report_dir, artifact_path="report")
    except Exception:
        return
