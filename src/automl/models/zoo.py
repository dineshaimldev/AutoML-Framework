from __future__ import annotations

from typing import Callable

import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_logistic_regression(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    C = trial.suggest_float("logreg_C", 1e-3, 1e2, log=True)
    solver = trial.suggest_categorical("logreg_solver", ["lbfgs", "liblinear"])
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000,
        random_state=random_state,
    )


def build_random_forest(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    n_estimators = trial.suggest_int("rf_n_estimators", 50, 400)
    max_depth = trial.suggest_int("rf_max_depth", 2, 32)
    min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )


def build_gradient_boosting(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    n_estimators = trial.suggest_int("gb_n_estimators", 50, 300)
    learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 0.3, log=True)
    max_depth = trial.suggest_int("gb_max_depth", 2, 10)
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def build_svm(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    C = trial.suggest_float("svm_C", 1e-3, 1e2, log=True)
    kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
    return SVC(
        C=C,
        kernel=kernel,
        probability=True,  # needed for ROC-AUC / predict_proba later
        random_state=random_state,
    )


# Registry mapping config model names -> builder functions.
# Adding a new model later means: write a build_x function, add one line here.
MODEL_REGISTRY: dict[str, Callable[[optuna.Trial, int], BaseEstimator]] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
    "svm": build_svm,
}


def get_model_builder(model_name: str) -> Callable[[optuna.Trial, int], BaseEstimator]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
