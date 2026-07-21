"""FastAPI app for serving the best trained model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from automl.models.persistence import load_pipeline

app = FastAPI(title="AutoML Framework API", version="0.1.0")
MODEL_PATH = Path("models_saved/best_model.joblib")


def _model_path_exists() -> bool:
    path = Path(MODEL_PATH)
    return path.exists()


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature values keyed by column name")


@app.get("/health")
def health() -> dict[str, Any]:
    model_loaded = _model_path_exists()
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if not _model_path_exists():
        raise HTTPException(status_code=404, detail="No trained model found.")

    pipeline = load_pipeline(str(MODEL_PATH))
    return {
        "model_name": pipeline.model_name,
        "expected_features": pipeline.feature_names_raw,
        "metrics": pipeline.metrics,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if not _model_path_exists():
        raise HTTPException(status_code=404, detail="No trained model found.")

    pipeline = load_pipeline(str(MODEL_PATH))
    feature_frame = pd.DataFrame([request.features])
    missing_features = [
        feature for feature in pipeline.feature_names_raw if feature not in feature_frame.columns
    ]
    if missing_features:
        raise HTTPException(
            status_code=422, detail=f"Missing required features: {missing_features}"
        )

    feature_frame = feature_frame[pipeline.feature_names_raw]
    transformed_features = pipeline.preprocessor.transform(feature_frame)
    prediction = pipeline.model.predict(transformed_features)[0]
    probability = (
        pipeline.model.predict_proba(transformed_features)[0, 1]
        if hasattr(pipeline.model, "predict_proba")
        else None
    )
    return {
        "prediction": int(prediction),
        "probability": float(probability) if probability is not None else 0.0,
    }
