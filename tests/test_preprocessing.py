from pathlib import Path

import numpy as np
import pytest

from automl.config import DataConfig, PreprocessingConfig
from automl.data.loader import load_and_split
from automl.preprocessing.pipeline import build_preprocessor, detect_column_types

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample.csv"


@pytest.fixture
def split():
    data_config = DataConfig(path=str(SAMPLE_DATA_PATH), target_column="Churn")
    return load_and_split(data_config)


def test_detect_column_types_separates_numeric_and_categorical(split):
    column_types = detect_column_types(split.X_train)
    assert "TotalCharges" in column_types.numeric
    assert "gender" in column_types.categorical
    assert set(column_types.numeric) | set(column_types.categorical) == set(split.X_train.columns)


def test_preprocessor_handles_missing_values_without_error(split):
    column_types = detect_column_types(split.X_train)
    config = PreprocessingConfig()
    preprocessor = build_preprocessor(column_types, config)

    transformed = preprocessor.fit_transform(split.X_train)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(dense).any()


def test_preprocessor_transform_is_consistent_between_train_and_test(split):
    column_types = detect_column_types(split.X_train)
    config = PreprocessingConfig()
    preprocessor = build_preprocessor(column_types, config)

    preprocessor.fit(split.X_train)
    train_transformed = preprocessor.transform(split.X_train)
    test_transformed = preprocessor.transform(split.X_test)

    assert train_transformed.shape[1] == test_transformed.shape[1]


def test_preprocessor_onehot_expands_categorical_columns(split):
    column_types = detect_column_types(split.X_train)
    config = PreprocessingConfig(encode_categorical="onehot")
    preprocessor = build_preprocessor(column_types, config)

    transformed = preprocessor.fit_transform(split.X_train)
    raw_width = len(column_types.numeric) + len(column_types.categorical)
    assert transformed.shape[1] > raw_width
