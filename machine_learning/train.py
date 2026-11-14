"""Machine learning training utilities for ForgetLab.

This module trains baseline models and stores evaluation metrics in a reusable
format for comparison with unlearned models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from machine_learning.preprocessing import build_preprocessing_pipeline, detect_target_task, prepare_features


def _safe_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    """Split data while handling small remaining datasets after forgetting rows.

    scikit-learn requires at least two samples per class when stratification is
    used. For very small datasets or after removing selected records, this can
    fail even though the data is otherwise valid. In that case we fall back to a
    normal split while keeping a reproducible random state.
    """
    if stratify and y.nunique() >= 2 and min(y.value_counts().to_numpy()) >= 2:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_training_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    train_indices=None,
    test_indices=None,
):
    """Return a validated explicit split or the reproducible uploaded-CSV split."""
    X, y = prepare_features(df, target_column)
    task_type = detect_target_task(y)
    if train_indices is not None or test_indices is not None:
        if train_indices is None or test_indices is None:
            raise ValueError("Both training and held-out test indices are required.")
        train_indices = list(train_indices)
        test_indices = list(test_indices)
        usable_indices = set(y.index)
        if len(set(train_indices)) != len(train_indices) or len(set(test_indices)) != len(test_indices):
            raise ValueError("Training and test indices must not contain duplicates.")
        if not set(train_indices).issubset(set(df.index)) or not set(test_indices).issubset(set(df.index)):
            raise ValueError("Training and test indices must belong to the active dataset.")
        train_order = [index for index in train_indices if index in usable_indices]
        test_order = [index for index in test_indices if index in usable_indices]
        train_set = set(train_order)
        test_set = set(test_order)
        if train_set & test_set or train_set | test_set != usable_indices:
            raise ValueError("Training and held-out test indices must partition all rows with a usable target.")
        return X.loc[train_order], X.loc[test_order], y.loc[train_order], y.loc[test_order]
    return _safe_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=task_type.endswith("classification"),
    )


def train_model_on_split(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit on the supplied training rows and evaluate the unchanged test rows."""
    task_type = detect_target_task(y_train)
    if X_train.empty:
        raise ValueError("At least one training record must remain.")
    if task_type.endswith("classification") and y_train.nunique() < 2:
        raise ValueError("At least two target classes must remain in the training records.")
    estimator = _build_estimator(model_name, task_type, random_state)

    preprocessor = build_preprocessing_pipeline(X_train, scale_numeric=model_name.startswith(("sgd_", "logistic", "linear_")))
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    estimator = pipeline.named_steps["model"]
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coefficients = np.abs(estimator.coef_)
        importance_values = coefficients.mean(axis=0) if coefficients.ndim > 1 else coefficients
    else:
        importance_values = np.zeros(len(transformed_names))
    feature_importance = sorted(
        [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(transformed_names, importance_values)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:20]

    class_labels = pipeline.classes_.tolist() if task_type.endswith("classification") else []
    positive_label = sorted(y_train.unique().tolist())[-1] if task_type == "binary_classification" else None
    confusion = None
    roc_payload = None
    precision_recall_payload = None
    actual_vs_predicted = None
    residuals_payload = None
    if task_type.endswith("classification"):
        average = "binary" if task_type == "binary_classification" else "weighted"
        metric_options = {"average": average, "zero_division": 0}
        if average == "binary":
            metric_options["pos_label"] = positive_label
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, **metric_options)),
            "recall": float(recall_score(y_test, predictions, **metric_options)),
            "f1": float(f1_score(y_test, predictions, **metric_options)),
            "roc_auc": None,
        }
        confusion = confusion_matrix(y_test, predictions, labels=class_labels).tolist()
        try:
            probabilities = pipeline.predict_proba(X_test)
            if task_type == "binary_classification" and y_test.nunique() == 2:
                positive_probability = probabilities[:, class_labels.index(positive_label)]
                false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
                    y_test, positive_probability, pos_label=positive_label
                )
                metrics["roc_auc"] = float(roc_auc_score(y_test, positive_probability))
                roc_payload = {
                    "false_positive_rate": false_positive_rate.astype(float).tolist(),
                    "true_positive_rate": true_positive_rate.astype(float).tolist(),
                    "thresholds": [float(value) if np.isfinite(value) else None for value in roc_thresholds],
                }
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                    y_test, positive_probability, pos_label=positive_label
                )
                precision_recall_payload = {
                    "precision": precision_curve.astype(float).tolist(),
                    "recall": recall_curve.astype(float).tolist(),
                    "thresholds": [float(value) if np.isfinite(value) else None for value in pr_thresholds],
                }
            elif set(class_labels).issubset(set(y_test.unique())):
                metrics["roc_auc"] = float(roc_auc_score(
                    y_test, probabilities, labels=class_labels, multi_class="ovr", average="weighted"
                ))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        actual = np.asarray(y_test, dtype=float)
        predicted = np.asarray(predictions, dtype=float)
        residuals = actual - predicted
        metrics = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "mse": float(mean_squared_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        }
        actual_vs_predicted = {"actual": actual.astype(float).tolist(), "predicted": predicted.astype(float).tolist()}
        residuals_payload = {"predicted": predicted.astype(float).tolist(), "residuals": residuals.astype(float).tolist()}

    return {
        "model": pipeline,
        "model_name": model_name,
        "task_type": task_type,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "class_labels": [str(label) for label in class_labels],
        "positive_label": str(positive_label) if positive_label is not None else None,
        "confusion_matrix": confusion,
        "roc_curve": roc_payload,
        "precision_recall_curve": precision_recall_payload,
        "actual_vs_predicted": actual_vs_predicted,
        "residuals": residuals_payload,
        "feature_importance": feature_importance,
    }


def _build_estimator(model_name: str, task_type: str, random_state: int):
    """Resolve a supported estimator and reject classifier/regressor mismatches."""
    if task_type.endswith("classification"):
        estimators = {
            "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            "random_forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=random_state),
            "random_forest_classifier": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=random_state),
            "sgd_classifier": lambda: SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", random_state=random_state),
        }
    else:
        estimators = {
            "linear_regression": LinearRegression,
            "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=random_state),
            "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=random_state),
            "sgd_regressor": lambda: SGDRegressor(max_iter=1000, random_state=random_state),
        }
    if model_name not in estimators:
        raise ValueError(f"Model '{model_name}' is not supported for {task_type}.")
    constructor = estimators[model_name]
    return constructor() if callable(constructor) else constructor()


def train_model(
    model_name: str,
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    train_indices=None,
    test_indices=None,
) -> Dict[str, Any]:
    """Train a model on one reproducible split and return its evaluation artifacts."""
    X_train, X_test, y_train, y_test = split_training_data(
        df,
        target_column,
        test_size=test_size,
        random_state=random_state,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return train_model_on_split(model_name, X_train, y_train, X_test, y_test, random_state)


def save_model(model, model_path: str) -> str:
    """Persist a trained model to disk using joblib.

    Args:
        model: Fitted sklearn pipeline.
        model_path: Destination path on disk.

    Returns:
        The saved model path.
    """
    directory = Path(model_path).parent
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(model_path: str):
    """Load a trained model from disk.

    Args:
        model_path: File path to a saved model.

    Returns:
        The loaded sklearn model object.
    """
    return joblib.load(model_path)
