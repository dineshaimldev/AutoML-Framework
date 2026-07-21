"""Load a dataset from CSV and split it into train/test sets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automl.config import DataConfig


@dataclass
class DatasetSplit:
    """Holds a train/test split as separate feature frames and target series."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


class DataValidationError(ValueError):
    """Raised when the input dataset fails a basic sanity check."""


def clean_known_quirks(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset-specific cleanup that must happen identically at both
    training time and prediction time, so the model never sees a different
    shape of data than it was trained on.

    This function is called from both `load_dataset` (training path) and
    `predict_from_csv` in cli.py (inference path) -- keeping the cleaning
    logic in exactly one place avoids the two paths silently drifting apart.
    """
    df = df.copy()

    # customerID is a unique identifier with no predictive signal -- drop it.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges is really numeric but stored as text with a few blank
    # strings. errors="coerce" turns those blanks into NaN, which the
    # preprocessing pipeline's imputer will later fill.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def load_dataset(data_config: DataConfig) -> pd.DataFrame:
    """Read the CSV at `data_config.path`, clean known quirks, return a DataFrame."""
    path = Path(data_config.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Check the `data.path` value in your config."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise DataValidationError(f"Dataset at '{path}' is empty.")

    df = clean_known_quirks(df)

    # sklearn classifiers need a numeric target, not "Yes"/"No" strings.
    if data_config.target_column in df.columns and df[data_config.target_column].dtype == object:
        unique_vals = set(df[data_config.target_column].dropna().unique())
        if unique_vals <= {"Yes", "No"}:
            df[data_config.target_column] = df[data_config.target_column].map(
                {"Yes": 1, "No": 0}
            )

    # --- Generic validation (applies to any dataset) ---

    if data_config.target_column not in df.columns:
        raise DataValidationError(
            f"Target column '{data_config.target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    if df[data_config.target_column].isna().any():
        n_missing = int(df[data_config.target_column].isna().sum())
        raise DataValidationError(
            f"Target column '{data_config.target_column}' has {n_missing} missing value(s). "
            "Rows with a missing label cannot be used for supervised training — "
            "drop or impute them before running the pipeline."
        )

    if df[data_config.target_column].nunique() < 2:
        raise DataValidationError(
            f"Target column '{data_config.target_column}' has fewer than 2 distinct classes; "
            "classification requires at least 2."
        )

    return df


def split_dataset(df: pd.DataFrame, data_config: DataConfig) -> DatasetSplit:
    """Split a loaded DataFrame into stratified train/test feature/target sets."""
    y = df[data_config.target_column]
    X = df.drop(columns=[data_config.target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
        stratify=y,
    )

    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def load_and_split(data_config: DataConfig) -> DatasetSplit:
    """Convenience wrapper: load the CSV and immediately split it."""
    df = load_dataset(data_config)
    return split_dataset(df, data_config)