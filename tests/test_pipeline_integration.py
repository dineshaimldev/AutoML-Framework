from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl.cli import predict_from_csv, run_pipeline
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
from automl.models.persistence import TrainedPipeline, load_pipeline, save_pipeline


def _write_synthetic_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65, 20, 30, 40, 50, 60, 25, 35, 45, 55, 65, 20, 30, 40, 50, 60],
            "income": [
                30000,
                40000,
                50000,
                60000,
                70000,
                32000,
                42000,
                52000,
                62000,
                72000,
                31000,
                41000,
                51000,
                61000,
                71000,
                33000,
                43000,
                53000,
                63000,
                73000,
            ],
            "city": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "target": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        }
    )
    df.to_csv(path, index=False)


def test_full_pipeline_runs_end_to_end(tmp_path):
    data_path = tmp_path / "synthetic.csv"
    _write_synthetic_dataset(data_path)

    config_path = tmp_path / "config.yaml"
    config = AutoMLConfig(
        data=DataConfig(
            path=str(data_path), target_column="target", test_size=0.2, random_state=42
        ),
        task=TaskConfig(type="classification"),
        preprocessing=PreprocessingConfig(scale_numeric=True, encode_categorical="onehot"),
        models=[ModelEntry(name="logistic_regression", enabled=True)],
        optimization=OptimizationConfig(
            n_trials=2, timeout_seconds=30, cv_folds=2, metric="accuracy"
        ),
        mlflow=MLflowConfig(enabled=False),
        reporting=ReportingConfig(
            output_dir=str(tmp_path / "reports"), formats=["html"], include_shap=False
        ),
        random_seed=42,
    )
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    run_pipeline(str(config_path))

    report_dir = next((tmp_path / "reports").glob("run_*"))
    report_path = report_dir / "report.html"
    model_path = Path("models_saved") / "best_model.joblib"

    assert report_path.exists(), "html report should be generated"
    assert model_path.exists(), "model artifact should be saved"

    report_html = report_path.read_text(encoding="utf-8")
    assert "AutoML Evaluation Report" in report_html


def test_predict_from_csv_writes_predictions(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "model.joblib"

    pd.DataFrame(
        {
            "age": [25, 35],
            "income": [30000, 40000],
            "city": ["A", "B"],
        }
    ).to_csv(input_path, index=False)

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age", "income"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(
        pd.DataFrame(
            {
                "age": [25, 35, 45, 55],
                "income": [30000, 40000, 50000, 60000],
                "city": ["A", "B", "A", "B"],
            }
        ),
        [0, 0, 1, 1],
    )

    trained_pipeline = TrainedPipeline(
        preprocessor=preprocessor,
        model=model,
        feature_names_raw=["age", "income", "city"],
        feature_names_transformed=["num__age", "num__income", "cat__A", "cat__B"],
        model_name="logistic_regression",
        metrics={"accuracy": 1.0},
    )
    save_pipeline(trained_pipeline, str(model_path))

    predict_from_csv(str(model_path), str(input_path), str(output_path))

    predictions = pd.read_csv(output_path)
    assert {"prediction"}.issubset(predictions.columns)
    assert len(predictions) == 2


def test_saved_pipeline_tracks_dataset_hash(tmp_path):
    model_path = tmp_path / "model.joblib"
    dataframe = pd.DataFrame({"age": [25, 35], "income": [30000, 40000], "city": ["A", "B"]})
    dataframe.to_csv(tmp_path / "train.csv", index=False)

    trained_pipeline = TrainedPipeline(
        preprocessor=ColumnTransformer(
            [
                ("num", StandardScaler(), ["age", "income"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
            ],
            remainder="drop",
        ),
        model=LogisticRegression(max_iter=1000),
        feature_names_raw=["age", "income", "city"],
        feature_names_transformed=["num__age", "num__income", "cat__A", "cat__B"],
        model_name="logistic_regression",
        metrics={"accuracy": 1.0},
    )
    save_pipeline(trained_pipeline, str(model_path))

    loaded_pipeline = load_pipeline(str(model_path))
    assert loaded_pipeline.dataset_hash
