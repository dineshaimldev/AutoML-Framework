from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from automl.config import PreprocessingConfig


@dataclass
class ColumnTypes:
    numeric: list[str]
    categorical: list[str]


def detect_column_types(X: pd.DataFrame) -> ColumnTypes:
    """Split columns into numeric vs categorical based on dtype."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return ColumnTypes(numeric=numeric_cols, categorical=categorical_cols)


def build_preprocessor(
    column_types: ColumnTypes, preprocessing_config: PreprocessingConfig
) -> ColumnTransformer:
    """Construct a ColumnTransformer that imputes (+scales/encodes) each column group."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=preprocessing_config.numeric_impute_strategy)),
    ]
    if preprocessing_config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    if preprocessing_config.encode_categorical == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore")
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy=preprocessing_config.categorical_impute_strategy),
            ),
            ("encoder", encoder),
        ]
    )

    transformers = []
    if column_types.numeric:
        transformers.append(("numeric", numeric_pipeline, column_types.numeric))
    if column_types.categorical:
        transformers.append(("categorical", categorical_pipeline, column_types.categorical))

    if not transformers:
        raise ValueError("No numeric or categorical columns found to preprocess.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
