"""Machine unlearning experiments for ForgetLab.

This module contains a strict exact retraining baseline and an approximate
unlearning experiment based on a stochastic gradient classifier. The exact
retraining approach is the scientifically grounded baseline; the approximate
method is treated as an experimental comparison, not a guaranteed privacy
removal guarantee.
"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from machine_learning.preprocessing import detect_target_task
from machine_learning.train import train_model, train_model_on_split


def _evaluation_artifacts(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep JSON-safe held-out diagnostics from a trained reference model."""
    return {
        "training_count": len(result["X_train"]),
        "testing_count": len(result["X_test"]),
        "test_indices": [int(index) for index in result["X_test"].index],
        "class_labels": result.get("class_labels", []),
        "confusion_matrix": result.get("confusion_matrix", []),
        "roc_curve": result.get("roc_curve"),
        "precision_recall_curve": result.get("precision_recall_curve"),
        "feature_importance": result.get("feature_importance", []),
        "actual_vs_predicted": result.get("actual_vs_predicted"),
        "residuals": result.get("residuals"),
    }


def _validate_training_indices(forget_indices: Iterable[int], training_indices: pd.Index) -> List[int]:
    """Reject empty, duplicate, or held-out selections before any model work."""
    selected = list(forget_indices)
    if not selected:
        raise ValueError("Select at least one training record to forget.")
    if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in selected):
        raise ValueError("Record indices must be integers.")
    if len(set(selected)) != len(selected):
        raise ValueError("A record can only be selected once.")
    training_index_set = set(training_indices)
    if any(index not in training_index_set for index in selected):
        raise ValueError("Only records in the training split can be selected for forgetting.")
    if len(selected) >= len(training_indices):
        raise ValueError("At least one training record must remain after forgetting.")
    return selected


def exact_retraining_unlearning(
    df: pd.DataFrame,
    target_column: str,
    forget_indices: Iterable[int],
    model_name: str = "logistic_regression",
    train_indices: Iterable[int] | None = None,
    test_indices: Iterable[int] | None = None,
) -> Dict[str, Any]:
    """Perform exact retraining after removing selected rows from the training set.

    This method is intentionally a reference baseline because it removes the
    data points from the learning objective and retrains the model on the
    remaining data. It is correct, but not efficient for large-scale ML systems.

    Args:
        df: Full dataset.
        target_column: Target label column.
        forget_indices: Row indices selected for forgetting.
        model_name: Name of the model to train.

    Returns:
        Dictionary containing original metrics, retrained metrics, and comparison
        summary information.
    """
    forget_index_list = list(forget_indices)
    original_result = train_model(
        model_name, df, target_column,
        train_indices=train_indices, test_indices=test_indices,
    )
    forget_index_list = _validate_training_indices(forget_index_list, original_result["X_train"].index)
    original_metrics = original_result["metrics"]

    retained_indices = original_result["X_train"].index.difference(forget_index_list)
    # Forgetting changes fit rows only; every comparison keeps the baseline test partition.
    retraining_started = time.perf_counter()
    retrained_result = train_model_on_split(
        model_name,
        original_result["X_train"].loc[retained_indices],
        original_result["y_train"].loc[retained_indices],
        original_result["X_test"],
        original_result["y_test"],
    )
    retraining_time = time.perf_counter() - retraining_started
    retrained_metrics = retrained_result["metrics"]

    task_type = original_result["task_type"]
    comparison_keys = ["mae", "mse", "rmse", "r2"] if task_type == "regression" else ["accuracy", "precision", "recall", "f1"]
    comparison = {
        f"{key}_delta": retrained_metrics[key] - original_metrics[key]
        for key in comparison_keys
        if original_metrics.get(key) is not None and retrained_metrics.get(key) is not None
    }
    comparison["removed_rows"] = len(forget_index_list)

    return {
        "original_metrics": original_metrics,
        "unlearned_metrics": retrained_metrics,
        "task_type": task_type,
        "model_name": model_name,
        "task_type": task_type,
        "original_evaluation": _evaluation_artifacts(original_result),
        "unlearned_evaluation": _evaluation_artifacts(retrained_result),
        "retraining_time": retraining_time,
        "comparison": comparison,
        "method": "exact_retraining_baseline",
    }


def approximate_unlearning_experiment(
    df: pd.DataFrame,
    target_column: str,
    forget_indices: Iterable[int],
    model_name: str = "logistic_regression",
    train_indices: Iterable[int] | None = None,
    test_indices: Iterable[int] | None = None,
) -> Dict[str, Any]:
    """Run a scientifically limited approximate unlearning experiment.

    The approach uses a linear SGD classifier trained in an online fashion. A
    negative scaled gradient update is applied to a copy of the training data
    for the forgotten records to approximate removing their contribution.
    This is not a formal guarantee of unlearning and should be reported as an
    experiment, not a privacy certificate.

    Args:
        df: Dataset to use for the approximate experiment.
        target_column: Target column to predict.
        forget_indices: Training examples selected for approximate forgetting.
        model_name: Model family used for the experiment.

    Returns:
        Metrics and notes describing the approximate unlearning outcome.
    """
    forget_index_list = list(forget_indices)
    task_type = detect_target_task(df[target_column])
    if task_type.endswith("classification"):
        if model_name != "sgd_classifier":
            raise ValueError("Approximate classification unlearning is supported only for SGD Classifier.")
    elif model_name != "sgd_regressor":
        raise ValueError("Approximate regression unlearning is supported only for SGD Regressor.")

    base_result = train_model(
        model_name, df, target_column,
        train_indices=train_indices, test_indices=test_indices,
    )
    forget_index_list = _validate_training_indices(forget_index_list, base_result["X_train"].index)
    original_metrics = base_result["metrics"]
    X_train = base_result["X_train"]
    X_test = base_result["X_test"]
    y_train = base_result["y_train"]
    y_test = base_result["y_test"]
    pipeline = base_result["model"]
    feature_pipeline = pipeline.named_steps["preprocessor"]
    X_train_processed = feature_pipeline.transform(X_train)
    X_test_processed = feature_pipeline.transform(X_test)
    approximate_model = deepcopy(pipeline.named_steps["model"])

    # One small gradient-ascent update approximates removing each sample's loss contribution.
    for index in forget_index_list:
        row_position = X_train.index.get_loc(index)
        sparse_row = X_train_processed[row_position:row_position + 1]
        row = sparse_row.toarray()[0] if hasattr(sparse_row, "toarray") else np.asarray(sparse_row)[0]
        step_size = 0.05 / (1.0 + float(np.dot(row, row)))
        target_value = y_train.loc[index]
        if task_type.endswith("classification"):
            margins = np.asarray(approximate_model.decision_function(sparse_row), dtype=float).reshape(-1)
            if approximate_model.coef_.shape[0] == 1:
                target_vector = np.asarray([float(target_value == approximate_model.classes_[1])])
            else:
                target_vector = np.asarray([float(target_value == label) for label in approximate_model.classes_])
            errors = expit(margins) - target_vector
            approximate_model.coef_ += step_size * (
                errors[:, None] * row[None, :] + approximate_model.alpha * approximate_model.coef_
            )
            approximate_model.intercept_ += step_size * errors
        else:
            prediction = float(approximate_model.predict(sparse_row)[0])
            error = prediction - float(target_value)
            approximate_model.coef_ += step_size * (
                error * row + approximate_model.alpha * approximate_model.coef_
            )
            approximate_model.intercept_ += step_size * error

    predictions = approximate_model.predict(X_test_processed)
    if task_type.endswith("classification"):
        average = "binary" if task_type == "binary_classification" else "weighted"
        metric_options = {"average": average, "zero_division": 0}
        if average == "binary":
            metric_options["pos_label"] = sorted(y_train.unique().tolist())[-1]
        approximate_metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, **metric_options)),
            "recall": float(recall_score(y_test, predictions, **metric_options)),
            "f1": float(f1_score(y_test, predictions, **metric_options)),
            "roc_auc": None,
        }
        try:
            probabilities = approximate_model.predict_proba(X_test_processed)
            if task_type == "binary_classification" and y_test.nunique() == 2:
                positive_label = sorted(y_train.unique().tolist())[-1]
                approximate_metrics["roc_auc"] = float(roc_auc_score(
                    y_test, probabilities[:, list(approximate_model.classes_).index(positive_label)]
                ))
            elif set(approximate_model.classes_).issubset(set(y_test.unique())):
                approximate_metrics["roc_auc"] = float(roc_auc_score(
                    y_test, probabilities, labels=approximate_model.classes_, multi_class="ovr", average="weighted"
                ))
        except ValueError:
            approximate_metrics["roc_auc"] = None
    else:
        actual = np.asarray(y_test, dtype=float)
        residual = actual - np.asarray(predictions, dtype=float)
        approximate_metrics = {
            "mae": float(mean_absolute_error(actual, predictions)),
            "mse": float(mean_squared_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "r2": float(r2_score(actual, predictions)) if len(actual) >= 2 else None,
        }

    return {
        "original_metrics": original_metrics,
        "approximate_metrics": approximate_metrics,
        "task_type": task_type,
        "model_name": model_name,
        "original_evaluation": _evaluation_artifacts(base_result),
        "notes": [
            "Approximate unlearning applies one bounded gradient-ascent step per selected record to reduce its loss contribution.",
            "This experiment does not provide a formal privacy or deletion guarantee.",
        ],
        "method": "approximate_sgd_gradient_ascent_experiment",
    }
