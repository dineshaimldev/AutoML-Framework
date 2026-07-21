"""End-to-end integration test: runs the full pipeline on a tiny synthetic
dataset and asserts every stage actually produced real output.

This is intentionally separate from the unit tests in other files — those
test each module in isolation, this one proves they work together.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from automl.config import (
    AutoMLConfig,
    DataConfig,
    MLflowConfig,
    ModelEntry,
    OptimizationConfig,
    PreprocessingConfig,
    ReportingConfig,
    TaskConfig,
)
from automl.data.loader import load_and_split
from automl.evaluation.metrics import fit_and_evaluate_best_model
from automl.evaluation.plots import plot_confusion_matrix, plot_roc_curve
from automl.optimization.search import pick_best_overall, search_all_models
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types
from automl.reporting.report_generator import generate_html_report


@pytest.fixture
def tiny_dataset(tmp_path):
    """A small, fast synthetic dataset -- just enough to exercise every stage."""
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3, n_redundant=0, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    df["category"] = np.random.RandomState(42).choice(["a", "b"], size=len(df))
    df["target"] = y

    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fast_config(tiny_dataset, tmp_path):
    """Minimal config: 1 model, 3 trials, so the whole test runs in seconds."""
    return AutoMLConfig(
        data=DataConfig(path=tiny_dataset, target_column="target", test_size=0.2, random_state=42),
        task=TaskConfig(type="classification"),
        preprocessing=PreprocessingConfig(),
        models=[ModelEntry(name="logistic_regression", enabled=True)],
        optimization=OptimizationConfig(n_trials=3, cv_folds=3, metric="roc_auc"),
        mlflow=MLflowConfig(enabled=False),
        reporting=ReportingConfig(output_dir=str(tmp_path / "reports"), formats=["html"]),
        random_seed=42,
    )


def test_full_pipeline_produces_real_outputs(fast_config):
    """Run every stage end-to-end and assert each one actually did its job."""
    # Stage 1: data loading + splitting
    split = load_and_split(fast_config.data)
    assert len(split.X_train) > 0
    assert len(split.X_test) > 0

    # Stage 2: preprocessing
    column_types = detect_column_types(split.X_train)
    preprocessor = build_preprocessor(column_types, fast_config.preprocessing)
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    assert X_train.shape[0] == len(split.X_train)

    # Stage 3: Optuna search
    results = search_all_models(X_train, split.y_train, fast_config)
    assert len(results) == 1
    best = pick_best_overall(results)
    assert 0.0 <= best.best_score <= 1.0

    # Stage 4: final evaluation
    evaluation = fit_and_evaluate_best_model(
        model_name=best.model_name,
        best_params=best.best_params,
        X_train=X_train,
        y_train=split.y_train,
        X_test=X_test,
        y_test=split.y_test,
        random_state=fast_config.random_seed,
    )
    assert 0.0 <= evaluation.roc_auc <= 1.0

    # Stage 5: plots actually get written to disk
    output_dir = fast_config.reporting.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cm_path = plot_confusion_matrix(evaluation, output_dir)
    roc_path = plot_roc_curve(evaluation, output_dir)
    assert Path(cm_path).exists()
    assert Path(roc_path).exists()

    # Stage 6: HTML report actually gets written to disk
    report_path = generate_html_report(
        output_dir=output_dir,
        dataset_path=fast_config.data.path,
        best_result=best,
        all_results=results,
        evaluation=evaluation,
        task_type=fast_config.task.type,
        optimization_metric=fast_config.optimization.metric,
        n_trials=fast_config.optimization.n_trials,
        cv_folds=fast_config.optimization.cv_folds,
        train_rows=len(split.X_train),
        test_rows=len(split.X_test),
        confusion_matrix_path=cm_path,
        roc_curve_path=roc_path,
    )
    assert Path(report_path).exists()
    assert Path(report_path).stat().st_size > 0
