"""Load and validate the YAML run configuration.

Keeping this as a strongly-typed Pydantic model (instead of a raw dict)
means bad configs fail fast with a clear error, instead of crashing
halfway through a training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: str
    target_column: str
    test_size: float = 0.2
    random_state: int = 42


class TaskConfig(BaseModel):
    type: Literal["classification", "regression"] = "classification"


class PreprocessingConfig(BaseModel):
    numeric_impute_strategy: Literal["mean", "median", "most_frequent"] = "median"
    categorical_impute_strategy: Literal["mean", "median", "most_frequent"] = "most_frequent"
    scale_numeric: bool = True
    encode_categorical: Literal["onehot", "ordinal"] = "onehot"


class ModelEntry(BaseModel):
    name: str
    enabled: bool = True


class OptimizationConfig(BaseModel):
    n_trials: int = 30
    timeout_seconds: int = 600
    cv_folds: int = 5
    sampler: Literal["tpe", "random"] = "tpe"
    pruner: Literal["median", "none"] = "median"
    metric: Literal["roc_auc", "accuracy", "f1"] = "roc_auc"


class MLflowConfig(BaseModel):
    enabled: bool = True
    experiment_name: str = "automl_framework_runs"
    tracking_uri: str = "./mlruns"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file_path: str | None = None


class ReportingConfig(BaseModel):
    output_dir: str = "reports"
    formats: list[str] = Field(default_factory=lambda: ["html"])
    include_shap: bool = True
    include_confusion_matrix: bool = True
    include_roc_curve: bool = True
    include_feature_importance: bool = True


class AutoMLConfig(BaseModel):
    data: DataConfig
    task: TaskConfig
    preprocessing: PreprocessingConfig
    models: list[ModelEntry]
    optimization: OptimizationConfig
    mlflow: MLflowConfig
    reporting: ReportingConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    random_seed: int = 42

    @property
    def enabled_model_names(self) -> list[str]:
        return [m.name for m in self.models if m.enabled]


def load_config(path: str | Path) -> AutoMLConfig:
    """Load a YAML config file and validate it into an AutoMLConfig."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return AutoMLConfig(**raw)
