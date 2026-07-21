from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


@dataclass
class TrainedPipeline:
    preprocessor: ColumnTransformer
    model: BaseEstimator
    feature_names_raw: list[str]  # columns expected in raw incoming data
    feature_names_transformed: list[str]  # columns after preprocessing (for SHAP/importance)
    model_name: str
    metrics: dict
    dataset_hash: str | None = None


def save_pipeline(pipeline: TrainedPipeline, output_path: str) -> str:
    """Serialize the full pipeline to disk with joblib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if pipeline.dataset_hash is None:
        pipeline.dataset_hash = "unknown"

    joblib.dump(pipeline, path)
    return str(path)


def load_pipeline(input_path: str) -> TrainedPipeline:
    """Load a previously saved pipeline. Raises a clear error if missing."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{path}'. Run `automl run --config <config>` first "
            "to train and save a model before starting the API."
        )
    return joblib.load(path)
