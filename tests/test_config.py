"""Tests for config loading and validation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from automl.config import AutoMLConfig, load_config

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "example_config.yaml"


def test_example_config_loads_successfully():
    config = load_config(EXAMPLE_CONFIG_PATH)
    assert isinstance(config, AutoMLConfig)
    assert config.task.type == "classification"


def test_enabled_model_names_excludes_disabled_models():
    config = load_config(EXAMPLE_CONFIG_PATH)
    assert "svm" not in config.enabled_model_names
    assert "random_forest" in config.enabled_model_names


def test_missing_config_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_invalid_task_type_is_rejected(tmp_path):
    bad_config = yaml.safe_load(open(EXAMPLE_CONFIG_PATH))
    bad_config["task"]["type"] = "not_a_real_task"

    bad_path = tmp_path / "bad_config.yaml"
    with open(bad_path, "w") as f:
        yaml.dump(bad_config, f)

    with pytest.raises(ValidationError):
        load_config(bad_path)


def test_optimization_defaults_are_sane():
    config = load_config(EXAMPLE_CONFIG_PATH)
    assert config.optimization.n_trials > 0
    assert 0 < config.data.test_size < 1
