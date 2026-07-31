"""FastAPI app for serving the best trained model and running background AutoML jobs."""

from __future__ import annotations

import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
from automl.models.persistence import load_pipeline
from automl.models.zoo import MODEL_REGISTRY
from automl.optimization.search import pick_best_overall, search_single_model
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types

app = FastAPI(title="AutoML Framework API", version="0.1.0")
MODEL_PATH = Path("models_saved/best_model.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://auto-ml-chat-ui.vercel.app",  # production frontend
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_path_exists() -> bool:
    return Path(MODEL_PATH).exists()


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature values keyed by column name")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _model_path_exists()}


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


# ---------------------------------------------------------------------------
# Background job system -- lets the frontend upload a CSV, kick off a real
# Optuna search in a background thread, and poll for progress/results.
# ---------------------------------------------------------------------------

# In-memory job store. Fine for a demo; a real production system would use
# Redis or a database so jobs survive a server restart.
_jobs: dict[str, dict] = {}

# Serializes heavy work so concurrent jobs don't silently compete for
# Render's single shared CPU core -- queued jobs wait cleanly instead
# of interleaving and making every job slower and harder to reason about.
_job_lock = threading.Lock()


def _run_job(
    job_id: str,
    csv_path: str,
    target_column: str,
    model_names: list[str],
    n_trials: int,
    cv_folds: int,
    metric: str,
) -> None:
    """Runs in a background thread so the HTTP request can return immediately.

    Acquires _job_lock before doing any real work, so only one job's Optuna
    search actually executes at a time. Other submitted jobs sit here
    waiting -- reported honestly as "queued" -- rather than silently
    interleaving and starving each other on a single shared CPU core.
    """
    _jobs[job_id]["status"] = "queued"
    _jobs[job_id]["progress"] = "Waiting for another job to finish..."

    with _job_lock:
        try:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = "Loading and splitting dataset..."

            config = AutoMLConfig(
                data=DataConfig(
                    path=csv_path, target_column=target_column, test_size=0.2, random_state=42
                ),
                task=TaskConfig(type="classification"),
                preprocessing=PreprocessingConfig(),
                models=[ModelEntry(name=m, enabled=True) for m in model_names],
                optimization=OptimizationConfig(
                    n_trials=n_trials, cv_folds=cv_folds, metric=metric
                ),
                mlflow=MLflowConfig(enabled=False),
                reporting=ReportingConfig(output_dir="reports", formats=["html"]),
                random_seed=42,
            )

            split = load_and_split(config.data)
            column_types = detect_column_types(split.X_train)
            preprocessor = build_preprocessor(column_types, config.preprocessing)
            X_train = preprocessor.fit_transform(split.X_train)
            X_test = preprocessor.transform(split.X_test)

            results = []
            fit_times: dict[str, float] = {}
            for model_name in model_names:
                _jobs[job_id]["progress"] = f"Searching {model_name}..."
                start = time.time()
                result = search_single_model(model_name, X_train, split.y_train, config)
                elapsed = time.time() - start
                fit_times[model_name] = elapsed
                results.append(result)

            best = pick_best_overall(results)

            _jobs[job_id]["progress"] = "Evaluating best model..."
            evaluation = fit_and_evaluate_best_model(
                model_name=best.model_name,
                best_params=best.best_params,
                X_train=X_train,
                y_train=split.y_train,
                X_test=X_test,
                y_test=split.y_test,
                random_state=config.random_seed,
            )

            # Shape the response exactly like the React app's ModelResult type expects
            leaderboard = sorted(results, key=lambda r: r.best_score, reverse=True)
            _jobs[job_id]["result"] = {
                "results": [
                    {
                        "name": r.model_name,
                        "score": r.best_score,
                        "params": r.best_params,
                        "fit_time": f"{fit_times[r.model_name]:.1f}s",
                    }
                    for r in leaderboard
                ],
                "metric": metric,
                "final_metrics": {
                    "accuracy": evaluation.accuracy,
                    "precision": evaluation.precision,
                    "recall": evaluation.recall,
                    "f1": evaluation.f1,
                    "roc_auc": evaluation.roc_auc,
                },
            }
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["progress"] = "Done"

        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),  # noqa: B008
    target_column: str = Form(...),  # noqa: B008
    models: str = Form(
        ...
    ),  # noqa: B008  -- comma-separated, e.g. "logistic_regression,random_forest"
    n_trials: int = Form(15),  # noqa: B008
    cv_folds: int = Form(5),  # noqa: B008
    metric: str = Form("roc_auc"),  # noqa: B008
):
    job_id = str(uuid.uuid4())
    n_trials = min(n_trials, 12)
    cv_folds = min(cv_folds, 3)
    if len(models.split(",")) > 2:
        raise HTTPException(
            status_code=422,
            detail="For this demo instance, please select at most 2 models per run.",
        )
    requested_models = [m.strip() for m in models.split(",")]
    unknown_models = [m for m in requested_models if m not in MODEL_REGISTRY]
    if unknown_models:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model(s): {unknown_models}. Available: {list(MODEL_REGISTRY.keys())}",
        )
    csv_path = str(Path(tempfile.gettempdir()) / f"{job_id}.csv")
    with open(csv_path, "wb") as f:
        f.write(await file.read())

    _jobs[job_id] = {
        "status": "queued",
        "progress": "Queued",
        "created": datetime.now().isoformat(),
    }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, csv_path, target_column, models.split(","), n_trials, cv_folds, metric),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}




@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]
