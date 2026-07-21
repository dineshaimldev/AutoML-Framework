"""Compute evaluation metrics for the best model found by the search.

Optuna's cross-validation never leaves us with one fitted model — each
fold trains a fresh copy internally. So the first job here is to actually
fit the best model's hyperparameters on the *full* training set, then
evaluate it once, honestly, on the held-out test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from automl.models.zoo import MODEL_REGISTRY


@dataclass
class EvaluationResult:
    model: BaseEstimator
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_mat: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray


def _instantiate_model_from_params(
    model_name: str, best_params: dict, random_state: int
) -> BaseEstimator:
    """Rebuild a model directly from its winning hyperparameters (no Optuna trial needed).

    Optuna prefixes each param with the model's short name (e.g. "rf_n_estimators"),
    so we strip that prefix to get the real sklearn constructor argument name.
    """
    prefix_map = {
        "logistic_regression": "logreg_",
        "random_forest": "rf_",
        "gradient_boosting": "gb_",
        "svm": "svm_",
    }
    prefix = prefix_map[model_name]
    clean_params = {k[len(prefix) :]: v for k, v in best_params.items() if k.startswith(prefix)}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=random_state, **clean_params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **clean_params)
    if model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=random_state, **clean_params)
    if model_name == "svm":
        from sklearn.svm import SVC

        return SVC(probability=True, random_state=random_state, **clean_params)

    raise ValueError(f"No instantiation logic for '{model_name}'")


def fit_and_evaluate_best_model(
    model_name: str,
    best_params: dict,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    random_state: int,
) -> EvaluationResult:
    """Fit the winning model on the full training set, evaluate once on the test set."""
    model = _instantiate_model_from_params(model_name, best_params, random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of the positive class

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return EvaluationResult(
        model=model,
        y_true=np.asarray(y_test),
        y_pred=y_pred,
        y_proba=y_proba,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
        confusion_mat=confusion_matrix(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
    )
