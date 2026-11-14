"""Model evaluation helpers for accuracy, fairness, and forgetting metrics."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a trained model using standard classification metrics.

    Args:
        model: Fitted sklearn model or pipeline.
        X_test: Test feature data.
        y_test: Actual outcomes.

    Returns:
        A dictionary of metrics including accuracy, precision, recall, and F1.
    """
    predictions = model.predict(X_test)
    positive_label = y_test.iloc[0] if len(y_test) else None
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="binary", pos_label=positive_label)) if positive_label is not None else 0.0,
        "recall": float(recall_score(y_test, predictions, average="binary", pos_label=positive_label)) if positive_label is not None else 0.0,
        "f1": float(f1_score(y_test, predictions, average="binary", pos_label=positive_label)) if positive_label is not None else 0.0,
    }
    return metrics


def forgetting_evaluation(original_metrics: Dict[str, float], unlearned_metrics: Dict[str, float]) -> Dict[str, float]:
    """Compute a simple forgetting delta for the comparison between original and unlearned models."""
    return {
        "accuracy_delta": float(unlearned_metrics.get("accuracy", 0.0) - original_metrics.get("accuracy", 0.0)),
        "precision_delta": float(unlearned_metrics.get("precision", 0.0) - original_metrics.get("precision", 0.0)),
        "recall_delta": float(unlearned_metrics.get("recall", 0.0) - original_metrics.get("recall", 0.0)),
        "f1_delta": float(unlearned_metrics.get("f1", 0.0) - original_metrics.get("f1", 0.0)),
    }
