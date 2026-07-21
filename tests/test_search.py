"""Tests for automl.optimization.search — kept small since Optuna trials are slow."""

from pathlib import Path

import pytest

from automl.config import AutoMLConfig, load_config
from automl.data.loader import load_and_split
from automl.optimization.search import (
    ModelSearchResult,
    pick_best_overall,
    search_all_models,
    search_single_model,
)
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "example_config.yaml"


@pytest.fixture
def fast_config() -> AutoMLConfig:
    """Load the example config but shrink trials so tests run in seconds, not minutes."""
    config = load_config(EXAMPLE_CONFIG_PATH)
    config.optimization.n_trials = 3
    config.optimization.cv_folds = 3
    return config


@pytest.fixture
def transformed_train_data(fast_config):
    split = load_and_split(fast_config.data)
    column_types = detect_column_types(split.X_train)
    preprocessor = build_preprocessor(column_types, fast_config.preprocessing)
    X_train_transformed = preprocessor.fit_transform(split.X_train)
    return X_train_transformed, split.y_train


def test_search_single_model_returns_valid_result(fast_config, transformed_train_data):
    X_train, y_train = transformed_train_data
    result = search_single_model("logistic_regression", X_train, y_train, fast_config)

    assert isinstance(result, ModelSearchResult)
    assert result.model_name == "logistic_regression"
    assert 0.0 <= result.best_score <= 1.0  # roc_auc is bounded [0, 1]
    assert "logreg_C" in result.best_params


def test_search_all_models_covers_every_enabled_model(fast_config, transformed_train_data):
    X_train, y_train = transformed_train_data
    results = search_all_models(X_train, y_train, fast_config)

    result_names = {r.model_name for r in results}
    assert result_names == set(fast_config.enabled_model_names)


def test_pick_best_overall_returns_highest_scoring_result():
    fake_results = [
        ModelSearchResult("model_a", best_score=0.80, best_params={}, study=None),
        ModelSearchResult("model_b", best_score=0.92, best_params={}, study=None),
        ModelSearchResult("model_c", best_score=0.75, best_params={}, study=None),
    ]
    best = pick_best_overall(fake_results)
    assert best.model_name == "model_b"


def test_pick_best_overall_raises_on_empty_list():
    with pytest.raises(ValueError, match="No search results"):
        pick_best_overall([])
