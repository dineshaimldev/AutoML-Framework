"""Load a dataset from CSV and split it into train/test sets.

Cleanup logic here is dataset-agnostic: it detects common real-world
quirks (identifier columns, text-encoded numbers, missing-value
placeholder strings, binary text targets) generically rather than
hardcoding column names from any one dataset. This is what lets the
framework actually run on a new CSV without requiring a code change
per dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from automl.config import DataConfig

logger = logging.getLogger(__name__)

# Common ways real-world datasets encode "no value" as text instead of
# leaving the cell truly blank. Matched case-insensitively after stripping
# whitespace.
_MISSING_VALUE_SENTINELS = ["?", "n/a", "na", "none", "null", "unknown", "-", "--", ""]


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


def clean_known_quirks(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Generic cleanup that must happen identically at training and
    prediction time, so the model never sees a different shape of data
    than it was trained on.

    Nothing here is hardcoded to one specific dataset's column names --
    every step below detects the quirk it fixes, so this works on any CSV:
      1. Normalize common missing-value placeholder strings to real NaN.
      2. Drop columns that are entirely missing (nothing to impute from).
      3. Drop obvious identifier columns (near-unique per row, no signal).
      4. Convert text-encoded numeric columns (e.g. "TotalCharges" stored
         as text with a few blank entries) into real numeric columns.
    """
    df = df.copy()
    n_rows = len(df)

    # 1. Normalize missing-value placeholder strings into real NaN, so the
    # imputer downstream actually sees them as missing instead of silently
    # treating them as their own valid category.
    for col in df.select_dtypes(include="object").columns:
        stripped = df[col].astype(str).str.strip().str.lower()
        is_sentinel = stripped.isin(_MISSING_VALUE_SENTINELS)
        if is_sentinel.any():
            logger.info(
                "Found %d missing-value placeholder(s) in column '%s' -- converting to NaN",
                int(is_sentinel.sum()),
                col,
            )
            df.loc[is_sentinel, col] = None

    for col in df.columns:
        if col == target_column:
            continue

        # 2. Drop columns that are entirely missing -- nothing for the
        # imputer to learn a fill value from, and zero predictive signal.
        if df[col].isna().all():
            logger.info("Dropping fully-empty column: %s", col)
            df = df.drop(columns=[col])
            continue

        # 3. Drop obvious identifier columns: a column where every value is
        # unique (or nearly every value, allowing for a few duplicate rows)
        # carries no predictive signal and just adds noise -- e.g. customerID,
        # transaction_id, row_number, etc, regardless of what it's named.
        if n_rows >= 20 and df[col].nunique(dropna=False) >= n_rows * 0.98:
            logger.info("Dropping likely identifier column: %s", col)
            df = df.drop(columns=[col])
            continue

        # 4. Convert text-encoded numeric columns: some datasets store a
        # genuinely numeric column as text, often because a few rows have
        # blank/placeholder strings (e.g. "TotalCharges" with " " entries).
        # If most non-null values convert cleanly to numbers, treat the
        # column as numeric and let a few unparseable values become NaN
        # (the preprocessing pipeline's imputer fills those in later).
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_original = df[col].notna().sum()
            non_null_converted = converted.notna().sum()
            if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.95:
                logger.info("Converting text-encoded numeric column: %s", col)
                df[col] = converted

    return df


def _encode_binary_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Encode a two-class text target into 0/1, regardless of what the
    actual class labels say (Yes/No, yes/no, True/False, Churned/Retained,
    Y/N, etc). Encoding is alphabetical for reproducibility, and the
    mapping is logged so it's always clear which label became 1.
    """
    if df[target_column].dtype != object:
        return df

    unique_vals = df[target_column].dropna().unique()
    if len(unique_vals) != 2:
        return df

    sorted_vals = sorted(unique_vals, key=lambda v: str(v).lower())
    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    logger.info("Encoding binary target '%s': %s", target_column, mapping)
    df[target_column] = df[target_column].map(mapping)
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

    df = clean_known_quirks(df, target_column=data_config.target_column)

    if data_config.target_column not in df.columns:
        raise DataValidationError(
            f"Target column '{data_config.target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    df = _encode_binary_target(df, data_config.target_column)

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
