from pathlib import Path

import pandas as pd
import pytest

from automl.config import DataConfig
from automl.data.loader import (
    DataValidationError,
    load_and_split,
    load_dataset,
    split_dataset,
)

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample.csv"


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(
        path=str(SAMPLE_DATA_PATH), target_column="Churn", test_size=0.2, random_state=42
    )


def test_load_dataset_reads_csv_successfully(data_config):
    df = load_dataset(data_config)
    assert isinstance(df, pd.DataFrame)
    assert "Churn" in df.columns
    assert len(df) > 0


def test_customer_id_column_is_dropped(data_config):
    df = load_dataset(data_config)
    assert "customerID" not in df.columns


def test_total_charges_is_converted_to_numeric(data_config):
    df = load_dataset(data_config)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_churn_target_is_converted_to_binary(data_config):
    df = load_dataset(data_config)
    assert set(df["Churn"].unique()) <= {0, 1}


def test_load_dataset_missing_file_raises_file_not_found():
    bad_config = DataConfig(path="does_not_exist.csv", target_column="Churn")
    with pytest.raises(FileNotFoundError):
        load_dataset(bad_config)


def test_load_dataset_missing_target_column_raises(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = tmp_path / "no_target.csv"
    df.to_csv(path, index=False)

    config = DataConfig(path=str(path), target_column="Churn")
    with pytest.raises(DataValidationError, match="not found in dataset columns"):
        load_dataset(config)


def test_split_dataset_produces_correct_proportions(data_config):
    df = load_dataset(data_config)
    split = split_dataset(df, data_config)

    total = len(df)
    assert len(split.X_train) + len(split.X_test) == total
    assert abs(len(split.X_test) / total - 0.2) < 0.02


def test_split_dataset_is_stratified(data_config):
    df = load_dataset(data_config)
    split = split_dataset(df, data_config)

    train_ratio = split.y_train.mean()
    test_ratio = split.y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_load_and_split_convenience_function(data_config):
    split = load_and_split(data_config)
    assert len(split.feature_names) > 0
    assert "Churn" not in split.feature_names
