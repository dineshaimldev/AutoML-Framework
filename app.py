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

st.set_page_config(page_title="Churn Intelligence Console", layout="wide", page_icon="\u25c6")

# ---------------------------------------------------------------------------
# Design system -- tokens pulled from the Figma export (light / orange brand)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&display=swap');

    :root {
        --bg: #ffffff;
        --bg-subtle: #fafafa;
        --bg-muted: #f5f5f5;
        --border: #e5e5e5;
        --border-strong: #d4d4d4;
        --text: #171717;
        --text-secondary: #525252;
        --text-muted: #a3a3a3;
        --orange: #f97316;
        --orange-dark: #ea580c;
        --orange-light: #fff7ed;
        --orange-mid: #fed7aa;
        --blue-info: #eff6ff;
        --blue-info-text: #1d4ed8;
        --blue-info-border: #bfdbfe;
        --green: #16a34a;
        --green-bg: #f0fdf4;
        --radius-sm: 6px;
        --radius: 10px;
        --radius-lg: 14px;
        --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.04);
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', system-ui, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: var(--bg-subtle);
        border-right: 1px solid var(--border);
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* -- hero -- */
    .console-hero {
        border-bottom: 1px solid var(--border);
        padding-bottom: 20px;
        margin-bottom: 28px;
    }
    .console-eyebrow {
        display: inline-block;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--orange-dark);
        text-transform: uppercase;
        background: var(--orange-light);
        border: 1px solid var(--orange-mid);
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .console-title {
        font-size: 30px;
        font-weight: 700;
        color: var(--text);
        margin: 0;
    }
    .console-subtitle {
        font-size: 14px;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* -- generic panel card (data preview, plots, leaderboard wrapper) -- */
    .panel-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 18px 20px;
        margin-bottom: 8px;
    }

    /* -- metric cards -- */
    .metric-card {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 16px 18px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'Inter', monospace;
        font-size: 26px;
        font-weight: 700;
        color: var(--text);
    }

    /* -- section labels -- */
    .section-label {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--text);
        border-left: 3px solid var(--orange);
        padding-left: 10px;
        margin: 32px 0 14px 0;
    }

    /* -- winner banner -- */
    .winner-banner {
        background: var(--orange-light);
        border: 1px solid var(--orange-mid);
        border-radius: var(--radius);
        padding: 14px 18px;
        font-size: 15px;
        font-weight: 600;
        color: var(--orange-dark);
        margin-bottom: 24px;
    }

    /* -- plot caption -- */
    .plot-caption {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-top: 6px;
        text-align: center;
    }

    /* -- streamlit widget overrides -- */
    .stButton > button {
        background: linear-gradient(135deg, var(--orange), var(--orange-dark));
        color: #ffffff;
        border: none;
        border-radius: var(--radius-sm);
        font-weight: 600;
        box-shadow: var(--shadow);
    }
    .stButton > button:hover {
        opacity: 0.92;
        color: #ffffff;
    }
    .stDownloadButton > button {
        background: var(--bg);
        color: var(--orange-dark);
        border: 1px solid var(--orange-mid);
        border-radius: var(--radius-sm);
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background: var(--orange-light);
        color: var(--orange-dark);
    }
    div[data-testid="stFileUploader"] {
        background: var(--bg-subtle);
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius);
        padding: 8px;
    }
    .stDataFrame {
        border-radius: var(--radius-sm);
        overflow: hidden;
    }
    div[data-testid="stStatusWidget"] {
        border-radius: var(--radius);
        border: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="console-hero">
        <div class="console-eyebrow">AutoML \u00b7 Model Search Console</div>
        <div class="console-title">Churn Intelligence</div>
        <div class="console-subtitle">Upload a dataset, pick a target column, and run a Bayesian search across the model zoo.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: configuration controls ---
with st.sidebar:
    st.markdown('<div class="section-label">01 \u00b7 Upload Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    st.markdown('<div class="section-label">02 \u00b7 Search Settings</div>', unsafe_allow_html=True)
    available_models = ["logistic_regression", "random_forest", "gradient_boosting", "svm"]
    selected_models = st.multiselect(
        "Models to try", available_models, default=["logistic_regression", "random_forest"]
    )
    n_trials = st.slider("Optuna trials per model", min_value=5, max_value=50, value=15)
    cv_folds = st.slider("Cross-validation folds", min_value=3, max_value=10, value=5)
    metric = st.selectbox("Optimization metric", ["roc_auc", "accuracy", "f1"])

    run_button = st.button("Run AutoML", type="primary", use_container_width=True)


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
    st.info("Upload a CSV from the sidebar to get started.")
    st.stop()

preview_df = pd.read_csv(uploaded_file)
st.markdown('<div class="section-label">Data Preview</div>', unsafe_allow_html=True)
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
    st.markdown(
        f'<div class="winner-banner">\u2713 Best model: {best.model_name}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Test Set Performance</div>', unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metrics_to_show = [
        ("Accuracy", evaluation.accuracy),
        ("Precision", evaluation.precision),
        ("Recall", evaluation.recall),
        ("F1 Score", evaluation.f1),
        ("ROC-AUC", evaluation.roc_auc),
    ]
    for col, (label, value) in zip(metric_cols, metrics_to_show):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Model Leaderboard</div>', unsafe_allow_html=True)
    leaderboard_df = pd.DataFrame(
        [{"model": r.model_name, f"{metric} (CV mean)": r.best_score} for r in
         sorted(results, key=lambda r: r.best_score, reverse=True)]
    )
    st.dataframe(leaderboard_df, use_container_width=True)

    st.markdown('<div class="section-label">Diagnostic Plots</div>', unsafe_allow_html=True)
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.image(cm_path, use_container_width=True)
        st.markdown('<div class="plot-caption">Confusion Matrix</div>', unsafe_allow_html=True)
    with plot_col2:
        st.image(roc_path, use_container_width=True)
        st.markdown('<div class="plot-caption">ROC Curve</div>', unsafe_allow_html=True)

    if fi_path:
        st.image(fi_path, use_container_width=True)
        st.markdown('<div class="plot-caption">Feature Importance</div>', unsafe_allow_html=True)
    if shap_path:
        st.image(shap_path, use_container_width=True)
        st.markdown('<div class="plot-caption">SHAP Summary</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    with open(report_path, "rb") as f:
        st.download_button(
            "Download full HTML report",
            f,
            file_name="automl_report.html",
            mime="text/html",
            use_container_width=True,
        )