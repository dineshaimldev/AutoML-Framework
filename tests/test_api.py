"""Tests for automl.api using FastAPI's TestClient — no real server needed."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from automl.config import load_config
from automl.data.loader import load_and_split
from automl.evaluation.metrics import fit_and_evaluate_best_model
from automl.models.persistence import TrainedPipeline, save_pipeline
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "example_config.yaml"


@pytest.fixture
def trained_model_path(tmp_path, monkeypatch):
    """Train a tiny real model and save it, then point the API at that saved file."""
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

    trained_pipeline = TrainedPipeline(
        preprocessor=preprocessor,
        model=evaluation.model,
        feature_names_raw=split.feature_names,
        feature_names_transformed=preprocessor.get_feature_names_out().tolist(),
        model_name="logistic_regression",
        metrics={"accuracy": evaluation.accuracy, "roc_auc": evaluation.roc_auc},
    )

    model_path = tmp_path / "test_model.joblib"
    save_pipeline(trained_pipeline, str(model_path))

    # redirect the API module's MODEL_PATH to our temp test model
    import automl.api as api_module

    monkeypatch.setattr(api_module, "MODEL_PATH", str(model_path))

    return split, model_path


@pytest.fixture
def client(trained_model_path):
    from automl.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_endpoint_returns_expected_features(client, trained_model_path):
    split, _ = trained_model_path
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert set(body["expected_features"]) == set(split.feature_names)


def test_predict_endpoint_returns_valid_prediction(client, trained_model_path):
    split, _ = trained_model_path
    sample_row = split.X_train.iloc[0].to_dict()

    response = client.post("/predict", json={"features": sample_row})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_endpoint_rejects_missing_features(client):
    response = client.post("/predict", json={"features": {"only_one_field": 123}})
    assert response.status_code == 422
    assert "Missing required features" in response.json()["detail"]
