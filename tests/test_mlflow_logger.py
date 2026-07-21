"""Tests for automl.tracking.mlflow_logger — uses a temp tracking URI so tests
never pollute your real mlruns/ folder or require network access.
"""

from pathlib import Path

import mlflow
import pytest

from automl.config import load_config
from automl.tracking.mlflow_logger import mlflow_run, setup_mlflow

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "example_config.yaml"


@pytest.fixture
def isolated_mlflow_config(tmp_path):
    config = load_config(EXAMPLE_CONFIG_PATH)
    config.mlflow.tracking_uri = str(tmp_path / "test_mlruns")
    config.mlflow.experiment_name = "test_experiment"
    return config


def test_setup_mlflow_creates_experiment(isolated_mlflow_config):
    setup_mlflow(isolated_mlflow_config)
    experiment = mlflow.get_experiment_by_name("test_experiment")
    assert experiment is not None


def test_mlflow_run_context_manager_starts_and_ends_run(isolated_mlflow_config):
    setup_mlflow(isolated_mlflow_config)

    with mlflow_run("test_run"):
        active_run = mlflow.active_run()
        assert active_run is not None
        assert active_run.info.run_name == "test_run"

    # after the context exits, there should be no active run
    assert mlflow.active_run() is None


def test_mlflow_run_ends_even_on_exception(isolated_mlflow_config):
    setup_mlflow(isolated_mlflow_config)

    with pytest.raises(ValueError):
        with mlflow_run("failing_run"):
            raise ValueError("simulated failure")

    # run should still be properly closed despite the exception
    assert mlflow.active_run() is None


def test_log_trial_as_nested_run_does_not_raise_on_mlflow_error(monkeypatch):
    class DummyTrial:
        number = 0
        params = {"foo": "bar"}

    def fail_start_run(*args, **kwargs):
        raise mlflow.exceptions.MlflowException("simulated MLflow failure")

    monkeypatch.setattr(mlflow, "start_run", fail_start_run)

    from automl.tracking import mlflow_logger

    mlflow_logger.log_trial_as_nested_run(DummyTrial(), "model", "roc_auc", 0.5)
