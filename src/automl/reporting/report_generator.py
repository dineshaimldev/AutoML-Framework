"""Render the final HTML evaluation report from a Jinja2 template."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from automl.evaluation.metrics import EvaluationResult
from automl.optimization.search import ModelSearchResult

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_html_report(
    output_dir: str,
    dataset_path: str,
    best_result: ModelSearchResult,
    all_results: list[ModelSearchResult],
    evaluation: EvaluationResult,
    task_type: str,
    optimization_metric: str,
    n_trials: int,
    cv_folds: int,
    train_rows: int,
    test_rows: int,
    confusion_matrix_path: str | None = None,
    roc_curve_path: str | None = None,
    feature_importance_path: str | None = None,
    shap_summary_path: str | None = None,
) -> str:
    """Render report_template.html with real run data and save it to disk.

    Plot paths are passed in as relative filenames (e.g. "confusion_matrix.png")
    because the report HTML and the PNG files are saved side-by-side in the
    same output_dir — that keeps the report portable if the whole folder
    is zipped up or moved.
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report_template.html")

    # Leaderboard sorted best-to-worst so the template can highlight the winner
    leaderboard = sorted(all_results, key=lambda r: r.best_score, reverse=True)

    def _basename_or_none(path: str | None) -> str | None:
        return Path(path).name if path else None

    html = template.render(
        run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dataset_path=dataset_path,
        best_model_name=best_result.model_name,
        best_params=best_result.best_params,
        leaderboard=leaderboard,
        accuracy=evaluation.accuracy,
        precision=evaluation.precision,
        recall=evaluation.recall,
        f1=evaluation.f1,
        roc_auc=evaluation.roc_auc,
        task_type=task_type,
        optimization_metric=optimization_metric,
        n_trials=n_trials,
        cv_folds=cv_folds,
        train_rows=train_rows,
        test_rows=test_rows,
        confusion_matrix_path=_basename_or_none(confusion_matrix_path),
        roc_curve_path=_basename_or_none(roc_curve_path),
        feature_importance_path=_basename_or_none(feature_importance_path),
        shap_summary_path=_basename_or_none(shap_summary_path),
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / "report.html"
    output_path.write_text(html, encoding="utf-8")

    return str(output_path)
