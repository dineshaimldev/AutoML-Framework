from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from automl.config import load_config
from automl.data.loader import load_and_split
from automl.evaluation.explainability import plot_shap_summary
from automl.evaluation.metrics import fit_and_evaluate_best_model
from automl.evaluation.plots import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
from automl.models.persistence import TrainedPipeline, load_pipeline, save_pipeline
from automl.optimization.search import pick_best_overall, search_all_models
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types
from automl.reporting.report_generator import generate_html_report
from automl.data.loader import load_and_split, clean_known_quirks
from automl.tracking.mlflow_logger import (
    log_final_evaluation,
    log_model_search_summary,
    log_report_artifacts,
    mlflow_run,
    setup_mlflow,
)

logger = logging.getLogger("automl.cli")


@contextmanager
def _noop_context():
    """A do-nothing context manager, used when MLflow is disabled so the
    `with run_context:` block still works without an if/else duplicating
    the entire pipeline body.
    """
    yield


def configure_logging(logging_config) -> None:
    """Configure console/file logging for the training CLI."""
    level_name = getattr(logging_config, "level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if logging_config.file_path:
        log_path = Path(logging_config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)


def _build_eval_output_dir(reporting_output_dir: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(reporting_output_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    configure_logging(config.logging)

    logger.info("Loaded config successfully")
    logger.info("Data path: %s", config.data.path)
    logger.info("Target column: %s", config.data.target_column)
    logger.info("Task type: %s", config.task.type)
    logger.info("Models enabled: %s", config.enabled_model_names)
    logger.info("Optuna trials: %s", config.optimization.n_trials)
    logger.info("Metric: %s", config.optimization.metric)

    logger.info("Loading and splitting dataset...")
    split = load_and_split(config.data)
    logger.info("Train rows: %s | Test rows: %s", len(split.X_train), len(split.X_test))
    logger.info("Feature columns: %s", split.feature_names)

    logger.info("Detecting column types and building preprocessor...")
    column_types = detect_column_types(split.X_train)
    logger.info("Numeric columns: %s", column_types.numeric)
    logger.info("Categorical columns: %s", column_types.categorical)

    preprocessor = build_preprocessor(column_types, config.preprocessing)
    transformed_train = preprocessor.fit_transform(split.X_train)
    transformed_test = preprocessor.transform(split.X_test)
    logger.info("Transformed train shape: %s", transformed_train.shape)
    logger.info("Transformed test shape: %s", transformed_test.shape)

    if config.mlflow.enabled:
        setup_mlflow(config)

    run_context = mlflow_run("automl_run") if config.mlflow.enabled else _noop_context()

    with run_context:
        logger.info("Running Optuna search across enabled models...")
        results = search_all_models(transformed_train, split.y_train, config)
        best = pick_best_overall(results)

        if config.mlflow.enabled:
            for result in results:
                log_model_search_summary(result, config.optimization.metric)

        logger.info("Best model overall: %s", best.model_name)
        logger.info("Best %s: %.4f", config.optimization.metric, best.best_score)
        logger.info("Best params: %s", best.best_params)

        logger.info("Fitting best model on full training set and evaluating on test set...")
        eval_output_dir = _build_eval_output_dir(config.reporting.output_dir)

        evaluation = fit_and_evaluate_best_model(
            model_name=best.model_name,
            best_params=best.best_params,
            X_train=transformed_train,
            y_train=split.y_train,
            X_test=transformed_test,
            y_test=split.y_test,
            random_state=config.random_seed,
        )

        logger.info("Accuracy: %.4f", evaluation.accuracy)
        logger.info("Precision: %.4f", evaluation.precision)
        logger.info("Recall: %.4f", evaluation.recall)
        logger.info("F1 score: %.4f", evaluation.f1)
        logger.info("ROC-AUC: %.4f", evaluation.roc_auc)

        if config.mlflow.enabled:
            log_final_evaluation(evaluation, best.model_name)

        logger.info("Generating evaluation plots...")
        feature_names = preprocessor.get_feature_names_out().tolist()

        cm_path = plot_confusion_matrix(evaluation, str(eval_output_dir))
        logger.info("Saved confusion matrix -> %s", cm_path)

        roc_path = plot_roc_curve(evaluation, str(eval_output_dir))
        logger.info("Saved ROC curve -> %s", roc_path)

        fi_path = plot_feature_importance(evaluation, feature_names, str(eval_output_dir))
        if fi_path:
            logger.info("Saved feature importance -> %s", fi_path)
        else:
            logger.info("Feature importance not available for %s", best.model_name)

        shap_path = None
        if config.reporting.include_shap:
            shap_path = plot_shap_summary(
                evaluation, transformed_train, feature_names, str(eval_output_dir)
            )
            if shap_path:
                logger.info("Saved SHAP summary -> %s", shap_path)

        if "html" in config.reporting.formats:
            logger.info("Generating HTML report...")
            report_path = generate_html_report(
                output_dir=str(eval_output_dir),
                dataset_path=config.data.path,
                best_result=best,
                all_results=results,
                evaluation=evaluation,
                task_type=config.task.type,
                optimization_metric=config.optimization.metric,
                n_trials=config.optimization.n_trials,
                cv_folds=config.optimization.cv_folds,
                train_rows=len(split.X_train),
                test_rows=len(split.X_test),
                confusion_matrix_path=cm_path,
                roc_curve_path=roc_path,
                feature_importance_path=fi_path,
                shap_summary_path=shap_path,
            )
            logger.info("Report saved -> %s", report_path)
            logger.info("Open it in a browser: file://%s", Path(report_path).resolve())

            if config.mlflow.enabled:
                log_report_artifacts(str(eval_output_dir))

        logger.info("Saving trained pipeline for serving...")
        dataset_path = Path(config.data.path)
        dataset_bytes = dataset_path.read_bytes() if dataset_path.exists() else b""
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

        trained_pipeline = TrainedPipeline(
            preprocessor=preprocessor,
            model=evaluation.model,
            feature_names_raw=split.feature_names,
            feature_names_transformed=feature_names,
            model_name=best.model_name,
            metrics={
                "accuracy": evaluation.accuracy,
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1": evaluation.f1,
                "roc_auc": evaluation.roc_auc,
            },
            dataset_hash=dataset_hash,
        )
        model_path = save_pipeline(trained_pipeline, "models_saved/best_model.joblib")
        logger.info("Saved -> %s", model_path)

        if config.mlflow.enabled:
            logger.info(
                "View experiment tracking: run `mlflow ui --backend-store-uri %s`",
                config.mlflow.tracking_uri,
            )


def predict_from_csv(model_path: str | os.PathLike[str], input_path: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> str:
    """Load a trained pipeline and write one prediction per input row."""
    pipeline = load_pipeline(str(model_path))
    input_frame = pd.read_csv(input_path)
    input_frame = clean_known_quirks(input_frame)

    missing_features = [
        feature for feature in pipeline.feature_names_raw if feature not in input_frame.columns
    ]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    ordered_frame = input_frame[pipeline.feature_names_raw]
    transformed_features = pipeline.preprocessor.transform(ordered_frame)
    predictions = pipeline.model.predict(transformed_features)

    output_frame = input_frame.copy()
    output_frame["prediction"] = predictions.astype(int)

    if hasattr(pipeline.model, "predict_proba"):
        probabilities = pipeline.model.predict_proba(transformed_features)[:, 1]
        output_frame["probability"] = probabilities

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(destination, index=False)
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(prog="automl", description="AutoML mini-framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the AutoML pipeline from a config file")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")

    predict_parser = subparsers.add_parser(
        "predict", help="Run batch prediction for a trained model"
    )
    predict_parser.add_argument("--model", required=True, help="Path to the trained pipeline")
    predict_parser.add_argument("--input", required=True, help="Path to the input CSV file")
    predict_parser.add_argument("--output", required=True, help="Path to the output CSV file")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args.config)
    elif args.command == "predict":
        predict_from_csv(args.model, args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
