"""Preprocessing helpers responsible for cleaning and transforming datasets.

This module converts categorical features to a numeric representation,
handles missing values, and prepares pipelines for model training and evaluation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(frame: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into features and target values.

    Args:
        frame: Input dataset.
        target_column: Name of the label column.

    Returns:
        A tuple of feature DataFrame and target Series.
    """
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    target_series = frame[target_column].copy()
    valid_targets = target_series.notna()
    if pd.api.types.is_object_dtype(target_series) or pd.api.types.is_string_dtype(target_series):
        normalized_target = target_series.astype("string").str.strip()
        valid_targets &= ~normalized_target.isin(["", "?"])
    target_series = target_series.loc[valid_targets]
    feature_frame = frame.drop(columns=[target_column]).loc[valid_targets].copy()
    return feature_frame, target_series


def detect_target_task(target: pd.Series) -> str:
    """Infer classification vs regression from dtype and target cardinality."""
    values = target.dropna()
    if values.empty:
        raise ValueError("The selected target contains no usable values.")
    cardinality = int(values.nunique())
    if cardinality < 2:
        raise ValueError("The selected target must contain at least two distinct values.")
    if pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_object_dtype(values.dtype) or pd.api.types.is_string_dtype(values.dtype) or isinstance(values.dtype, pd.CategoricalDtype):
        return "binary_classification" if cardinality == 2 else "multiclass_classification"
    if pd.api.types.is_numeric_dtype(values.dtype):
        if cardinality == 2:
            return "binary_classification"
        if pd.api.types.is_integer_dtype(values.dtype) and cardinality <= 10:
            return "multiclass_classification"
        return "regression"
    return "binary_classification" if cardinality == 2 else "multiclass_classification"


def get_supported_models(task_type: str) -> list[Dict[str, str]]:
    """Return model options whose estimator family matches the detected task."""
    if task_type.endswith("classification"):
        return [
            {"key": "logistic_regression", "name": "Logistic Regression"},
            {"key": "random_forest_classifier", "name": "Random Forest Classifier"},
            {"key": "sgd_classifier", "name": "SGD Classifier"},
        ]
    return [
        {"key": "linear_regression", "name": "Linear Regression"},
        {"key": "random_forest_regressor", "name": "Random Forest Regressor"},
        {"key": "sgd_regressor", "name": "SGD Regressor"},
    ]


def build_preprocessing_pipeline(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical columns.

    Args:
        X: Feature matrix.

    Returns:
        A configured preprocessing pipeline.
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "num",
                Pipeline(numeric_steps),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")
