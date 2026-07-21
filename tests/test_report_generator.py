"""Tests for automl.reporting.report_generator."""

from pathlib import Path

import pytest

from automl.config import load_config
from automl.data.loader import load_and_split
from automl.evaluation.metrics import fit_and_evaluate_best_model
from automl.optimization.search import ModelSearchResult
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types
from automl.reporting.report_generator import generate_html_report

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "example_config.yaml"


@pytest.fixture
def sample_report_inputs(tmp_path):
    config = load_config(EXAMPLE_CONFIG_PATH)
    split = load_and_split(config.data)
    column_types = detect_column_types(split.X_train)
    preprocessor = build_preprocessor(column_types, config.preprocessing)
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)

    evaluation = fit_and_evaluate_best_model(
        model_name="logistic_regression",
        best_params={"logreg_C": 1.0, "logreg_solver": "lbfgs"},
        X_train=X_train,
        y_train=split.y_train,
        X_test=X_test,
        y_test=split.y_test,
        random_state=config.random_seed,
    )

    best_result = ModelSearchResult(
        model_name="logistic_regression",
        best_score=evaluation.roc_auc,
        best_params={"logreg_C": 1.0, "logreg_solver": "lbfgs"},
        study=None,
    )

    return {
        "output_dir": str(tmp_path),
        "dataset_path": config.data.path,
        "best_result": best_result,
        "all_results": [best_result],
        "evaluation": evaluation,
        "task_type": config.task.type,
        "optimization_metric": config.optimization.metric,
        "n_trials": 3,
        "cv_folds": 3,
        "train_rows": len(split.X_train),
        "test_rows": len(split.X_test),
    }


def test_generate_html_report_creates_file(sample_report_inputs):
    report_path = generate_html_report(**sample_report_inputs)
    assert Path(report_path).exists()
    assert report_path.endswith("report.html")


def test_generated_report_contains_key_metrics(sample_report_inputs):
    report_path = generate_html_report(**sample_report_inputs)
    content = Path(report_path).read_text()

    assert "logistic_regression" in content
    assert "Accuracy" in content
    assert "ROC-AUC" in content


def test_generated_report_handles_missing_plots_gracefully(sample_report_inputs):
    # no plot paths passed -- should not crash or reference broken image tags
    report_path = generate_html_report(**sample_report_inputs)
    content = Path(report_path).read_text()
    assert "<img" not in content  # no plot paths given, so no <img> tags should render
