"""Tests for automl.evaluation.metrics."""
from pathlib import Path

import pytest

from automl.config import load_config
from automl.data.loader import load_and_split
from automl.evaluation.metrics import EvaluationResult, fit_and_evaluate_best_model
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "example_config.yaml"


@pytest.fixture
def prepared_data():
    config = load_config(EXAMPLE_CONFIG_PATH)
    split = load_and_split(config.data)
    column_types = detect_column_types(split.X_train)
    preprocessor = build_preprocessor(column_types, config.preprocessing)
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    return X_train, X_test, split.y_train, split.y_test, config


def test_fit_and_evaluate_returns_valid_result(prepared_data):
    X_train, X_test, y_train, y_test, config = prepared_data

    result = fit_and_evaluate_best_model(
        model_name="logistic_regression",
        best_params={"logreg_C": 1.0, "logreg_solver": "lbfgs"},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=config.random_seed,
    )

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert result.confusion_mat.shape == (2, 2)


def test_fit_and_evaluate_random_forest(prepared_data):
    X_train, X_test, y_train, y_test, config = prepared_data

    result = fit_and_evaluate_best_model(
        model_name="random_forest",
        best_params={"rf_n_estimators": 50, "rf_max_depth": 5, "rf_min_samples_split": 2},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=config.random_seed,
    )

    assert hasattr(result.model, "feature_importances_")
    assert len(result.y_pred) == len(y_test)