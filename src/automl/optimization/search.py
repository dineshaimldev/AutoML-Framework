from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from automl.config import AutoMLConfig
from automl.models.zoo import get_model_builder

# Quiet Optuna's default per-trial logging spam; we use our own logger.
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ModelSearchResult:
    model_name: str
    best_score: float
    best_params: dict
    study: optuna.Study


def _build_sampler(sampler_name: str, seed: int) -> optuna.samplers.BaseSampler:
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    return optuna.samplers.RandomSampler(seed=seed)


def _build_pruner(pruner_name: str) -> optuna.pruners.BasePruner:
    if pruner_name == "median":
        return optuna.pruners.MedianPruner()
    return optuna.pruners.NopPruner()


def search_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    config: AutoMLConfig,
) -> ModelSearchResult:
    """Run an Optuna study for one model type and return its best result."""
    builder = get_model_builder(model_name)
    cv = StratifiedKFold(
        n_splits=config.optimization.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    def objective(trial: optuna.Trial) -> float:
        try:
            model = builder(trial, config.random_seed)
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=config.optimization.metric,
                n_jobs=-1,
            )
            mean_score = float(scores.mean())

            if config.mlflow.enabled:
                from automl.tracking.mlflow_logger import log_trial_as_nested_run

                log_trial_as_nested_run(trial, model_name, config.optimization.metric, mean_score)

            return mean_score
        except Exception as exc:
            logger.warning("Trial %s for %s failed: %s", trial.number, model_name, exc)
            raise optuna.TrialPruned() from exc

    study = optuna.create_study(
        direction="maximize",
        sampler=_build_sampler(config.optimization.sampler, config.random_seed),
        pruner=_build_pruner(config.optimization.pruner),
        study_name=f"automl_{model_name}",
    )
    study.optimize(
        objective,
        n_trials=config.optimization.n_trials,
        timeout=config.optimization.timeout_seconds,
        show_progress_bar=False,
        n_jobs=-1,
    )

    return ModelSearchResult(
        model_name=model_name,
        best_score=study.best_value,
        best_params=study.best_params,
        study=study,
    )


def search_all_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    config: AutoMLConfig,
) -> list[ModelSearchResult]:
    """Run a separate Optuna study for every enabled model, return all results."""
    results = []
    for model_name in config.enabled_model_names:
        logger.info("Searching %s (%s trials)...", model_name, config.optimization.n_trials)
        result = search_single_model(model_name, X_train, y_train, config)
        logger.info("Best %s = %.4f", config.optimization.metric, result.best_score)
        results.append(result)
    return results


def pick_best_overall(results: list[ModelSearchResult]) -> ModelSearchResult:
    """Pick the single best model across all searched model types."""
    if not results:
        raise ValueError("No search results to pick from — was every model disabled?")
    return max(results, key=lambda r: r.best_score)
