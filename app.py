from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

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
from automl.evaluation.explainability import plot_shap_summary
from automl.evaluation.metrics import fit_and_evaluate_best_model
from automl.evaluation.plots import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
from automl.models.persistence import TrainedPipeline, save_pipeline
from automl.optimization.search import pick_best_overall, search_all_models
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types
from automl.reporting.report_generator import generate_html_report

st.set_page_config(page_title="AutoML Mini-Framework", layout="wide")
st.title("🤖 AutoML Mini-Framework")
st.caption("Upload a CSV, pick a target column, and let Optuna search models for you.")

# --- Sidebar: configuration controls ---
with st.sidebar:
    st.header("1. Upload data")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    st.header("2. Search settings")
    available_models = ["logistic_regression", "random_forest", "gradient_boosting", "svm"]
    selected_models = st.multiselect(
        "Models to try", available_models, default=["logistic_regression", "random_forest"]
    )
    n_trials = st.slider("Optuna trials per model", min_value=5, max_value=50, value=15)
    cv_folds = st.slider("Cross-validation folds", min_value=3, max_value=10, value=5)
    metric = st.selectbox("Optimization metric", ["roc_auc", "accuracy", "f1"])

    run_button = st.button("🚀 Run AutoML", type="primary", use_container_width=True)


def build_config_from_ui(target_column: str, data_path: str) -> AutoMLConfig:
    """Translate the sidebar widget values into the same AutoMLConfig the CLI uses."""
    return AutoMLConfig(
        data=DataConfig(path=data_path, target_column=target_column, test_size=0.2, random_state=42),
        task=TaskConfig(type="classification"),
        preprocessing=PreprocessingConfig(),
        models=[ModelEntry(name=m, enabled=(m in selected_models)) for m in available_models],
        optimization=OptimizationConfig(
            n_trials=n_trials, timeout_seconds=600, cv_folds=cv_folds, metric=metric
        ),
        mlflow=MLflowConfig(enabled=False),  # keep the demo UI simple, no MLflow noise here
        reporting=ReportingConfig(output_dir="reports", formats=["html"]),
        random_seed=42,
    )


# --- Main panel ---
if uploaded_file is None:
    st.info("Upload a CSV from the sidebar to get started. (Try the Telco Customer Churn dataset.)")
    st.stop()

preview_df = pd.read_csv(uploaded_file)
st.subheader("Data preview")
st.dataframe(preview_df.head(10), use_container_width=True)

target_column = st.selectbox("Target column (what you're predicting)", preview_df.columns.tolist())

if not run_button:
    st.info("Choose your settings in the sidebar, then click **Run AutoML**.")
    st.stop()

if not selected_models:
    st.error("Select at least one model in the sidebar before running.")
    st.stop()

# Save the uploaded file to a temp path so our existing loader (which expects a file path) can read it
with tempfile.TemporaryDirectory() as tmp_dir:
    data_path = str(Path(tmp_dir) / "uploaded.csv")
    preview_df.to_csv(data_path, index=False)

    config = build_config_from_ui(target_column, data_path)

    with st.status("Running AutoML pipeline...", expanded=True) as status:
        st.write("Loading and splitting dataset...")
        split = load_and_split(config.data)
        st.write(f"Train rows: {len(split.X_train)} | Test rows: {len(split.X_test)}")

        st.write("Detecting column types and preprocessing...")
        column_types = detect_column_types(split.X_train)
        preprocessor = build_preprocessor(column_types, config.preprocessing)
        X_train = preprocessor.fit_transform(split.X_train)
        X_test = preprocessor.transform(split.X_test)

        st.write(f"Searching {len(selected_models)} model type(s) with Optuna...")
        results = search_all_models(X_train, split.y_train, config)
        best = pick_best_overall(results)
        st.write(f"Best model: **{best.model_name}** ({metric} = {best.best_score:.4f})")

        st.write("Fitting best model and evaluating on test set...")
        evaluation = fit_and_evaluate_best_model(
            model_name=best.model_name,
            best_params=best.best_params,
            X_train=X_train,
            y_train=split.y_train,
            X_test=X_test,
            y_test=split.y_test,
            random_state=config.random_seed,
        )

        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        feature_names = preprocessor.get_feature_names_out().tolist()

        cm_path = plot_confusion_matrix(evaluation, report_dir)
        roc_path = plot_roc_curve(evaluation, report_dir)
        fi_path = plot_feature_importance(evaluation, feature_names, report_dir)
        shap_path = plot_shap_summary(evaluation, X_train, feature_names, report_dir)

        report_path = generate_html_report(
            output_dir=report_dir,
            dataset_path=uploaded_file.name,
            best_result=best,
            all_results=results,
            evaluation=evaluation,
            task_type=config.task.type,
            optimization_metric=metric,
            n_trials=n_trials,
            cv_folds=cv_folds,
            train_rows=len(split.X_train),
            test_rows=len(split.X_test),
            confusion_matrix_path=cm_path,
            roc_curve_path=roc_path,
            feature_importance_path=fi_path,
            shap_summary_path=shap_path,
        )

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
        )
        save_pipeline(trained_pipeline, "models_saved/best_model.joblib")

        status.update(label="Done!", state="complete")

    # --- Results ---
    st.success(f"Best model: **{best.model_name}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{evaluation.accuracy:.3f}")
    col2.metric("Precision", f"{evaluation.precision:.3f}")
    col3.metric("Recall", f"{evaluation.recall:.3f}")
    col4.metric("F1", f"{evaluation.f1:.3f}")
    col5.metric("ROC-AUC", f"{evaluation.roc_auc:.3f}")

    st.subheader("Model leaderboard")
    leaderboard_df = pd.DataFrame(
        [{"model": r.model_name, f"{metric} (CV mean)": r.best_score} for r in
         sorted(results, key=lambda r: r.best_score, reverse=True)]
    )
    st.dataframe(leaderboard_df, use_container_width=True)

    st.subheader("Diagnostic plots")
    plot_col1, plot_col2 = st.columns(2)
    plot_col1.image(cm_path, caption="Confusion Matrix")
    plot_col2.image(roc_path, caption="ROC Curve")
    if fi_path:
        st.image(fi_path, caption="Feature Importance")
    if shap_path:
        st.image(shap_path, caption="SHAP Summary")

    with open(report_path, "rb") as f:
        st.download_button(
            "📄 Download full HTML report", f, file_name="automl_report.html", mime="text/html"
        )