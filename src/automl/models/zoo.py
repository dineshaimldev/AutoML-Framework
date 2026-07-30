"""Model zoo: defines available models and their Optuna hyperparameter search spaces.

Each entry knows how to build a fresh, untrained sklearn estimator given
an Optuna `trial` object. The trial suggests values (e.g. trial.suggest_float),
and Optuna decides what to suggest next based on past trial results
(Bayesian optimization) — we just describe the search space per model here.
"""

from __future__ import annotations

from typing import Callable

import optuna
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


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
    n_estimators = trial.suggest_int("rf_n_estimators", 50, 150)
    max_depth = trial.suggest_int("rf_max_depth", 2, 20)
    min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,
    )


def build_gradient_boosting(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    n_estimators = trial.suggest_int("gb_n_estimators", 50, 150)
    learning_rate = trial.suggest_float("gb_learning_rate", 1e-3, 0.3, log=True)
    max_depth = trial.suggest_int("gb_max_depth", 2, 8)
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def build_svm(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    C = trial.suggest_float("svm_C", 1e-3, 1e2, log=True)
    kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
    base_svm = SVC(C=C, kernel=kernel, random_state=random_state)
    return CalibratedClassifierCV(base_svm, ensemble=False)


def build_xgboost(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    n_estimators = trial.suggest_int("xgb_n_estimators", 50, 150)
    learning_rate = trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True)
    max_depth = trial.suggest_int("xgb_max_depth", 2, 6)
    subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_knn(trial: optuna.Trial, random_state: int) -> BaseEstimator:
    n_neighbors = trial.suggest_int("knn_n_neighbors", 3, 50)
    weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
    p = trial.suggest_categorical("knn_p", [1, 2])
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)


MODEL_REGISTRY: dict[str, Callable[[optuna.Trial, int], BaseEstimator]] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
    "svm": build_svm,
    "xgboost": build_xgboost,
    "knn": build_knn,
}


def get_model_builder(model_name: str) -> Callable[[optuna.Trial, int], BaseEstimator]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
