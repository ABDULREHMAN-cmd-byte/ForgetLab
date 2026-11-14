"""Flask application entry point for ForgetLab.

The app factory initializes the database layer, registers the working routes, and
coordinates the real ML workflow: dataset upload, model training, exact
retraining, approximate experimentation, dashboard metrics, and export.
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from flask import Flask, current_app, jsonify, render_template, request, send_file

from config import CONFIG
from database import init_db
from machine_learning.dataset_manager import DatasetManager, detect_target_column, load_dataset, load_uci_adult_dataset, sanitize_filename, validate_csv_file
from machine_learning.evaluate import forgetting_evaluation
from machine_learning.report_generator import generate_experiment_summary, write_json_report
from machine_learning.preprocessing import detect_target_task, get_supported_models
from machine_learning.train import save_model, split_training_data, train_model
from machine_learning.unlearn import approximate_unlearning_experiment, exact_retraining_unlearning
from models_db import DatasetRecord, EvaluationResultRecord, ExperimentRecord, ModelRecord, UnlearningRequestRecord, db

app_state: Dict[str, Any] = {
    "dataset_path": None,
    "dataset_key": None,
    "dataset_name": None,
    "dataset_source": None,
    "dataframe": None,
    "target_column": "income",
    "split_indices": None,
    "partition_indices": None,
    "original_result": None,
    "unlearned_result": None,
    "unlearning_results": {},
    "forget_indices": [],
    "training_time": 0.0,
    "unlearning_time": 0.0,
}


def _default_dataset_path() -> str | None:
    """Return the official UCI training file when both source files exist."""
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    training_path = dataset_dir / "adult.data"
    testing_path = dataset_dir / "adult.test"
    return str(training_path) if training_path.is_file() and testing_path.is_file() else None


def _reset_model_state() -> None:
    """Clear metrics tied to the previous active dataset or target."""
    app_state["original_result"] = None
    app_state["unlearned_result"] = None
    app_state["unlearning_results"] = {}
    app_state["forget_indices"] = []
    app_state["unlearning_method"] = None
    app_state["training_time"] = 0.0
    app_state["unlearning_time"] = 0.0


def _activate_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    name: str,
    source: str,
    path: str,
    target_column: str,
    train_indices: Iterable[int] | None = None,
    test_indices: Iterable[int] | None = None,
) -> None:
    """Set dataset metadata and one shared split used by training and forgetting."""
    source_train_indices = list(train_indices) if train_indices is not None else None
    source_test_indices = list(test_indices) if test_indices is not None else None
    X_train, X_test, _, _ = split_training_data(
        df,
        target_column,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    app_state["dataset_path"] = path
    app_state["dataset_key"] = dataset_key
    app_state["dataset_name"] = name
    app_state["dataset_source"] = source
    app_state["dataframe"] = df.reset_index(drop=True).copy()
    app_state["target_column"] = target_column
    app_state["split_indices"] = {
        "train": [int(index) for index in X_train.index],
        "test": [int(index) for index in X_test.index],
    }
    app_state["partition_indices"] = {
        "train": source_train_indices or [int(index) for index in X_train.index],
        "test": source_test_indices or [int(index) for index in X_test.index],
    }
    _reset_model_state()


def _active_split() -> Dict[str, List[int]]:
    """Return the active split, rebuilding an uploaded-data split only if needed."""
    split = app_state.get("split_indices")
    if split is None:
        df = _get_active_dataframe()
        X_train, X_test, _, _ = split_training_data(df, app_state["target_column"])
        split = {
            "train": [int(index) for index in X_train.index],
            "test": [int(index) for index in X_test.index],
        }
        app_state["split_indices"] = split
    return split


def _available_datasets() -> List[Dict[str, Any]]:
    """List the official UCI pair and available uploaded CSVs without fabricating data."""
    options: List[Dict[str, Any]] = []
    default_path = _default_dataset_path()
    if default_path:
        testing_path = Path(default_path).with_name("adult.test")
        with Path(default_path).open(encoding="utf-8") as training_file:
            train_count = sum(1 for line in training_file if line.strip())
        with testing_path.open(encoding="utf-8") as testing_file:
            test_count = sum(1 for line in testing_file if line.strip() and not line.lstrip().startswith("|"))
        options.append({
            "key": "uci_adult",
            "name": "UCI Adult Census Income",
            "source": "UCI Machine Learning Repository · adult.data + adult.test",
            "rows": train_count + test_count,
            "columns": 15,
            "target_column": "income",
        })

    upload_dir = Path(CONFIG.UPLOAD_FOLDER)
    known_paths = {path.resolve() for path in upload_dir.glob("*.csv") if path.is_file()}
    try:
        known_paths.update(
            Path(record.file_path).resolve()
            for record in DatasetRecord.query.all()
            if Path(record.file_path).suffix.lower() == ".csv" and Path(record.file_path).is_file()
        )
    except RuntimeError:
        pass
    for path in sorted(known_paths):
        try:
            frame = load_dataset(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        target = detect_target_column(frame, "income")
        options.append({
            "key": f"upload:{path.name}",
            "name": path.name,
            "source": "Uploaded CSV",
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "target_column": target,
        })
    return options


def _ensure_default_dataset() -> pd.DataFrame:
    """Load both official UCI Adult files as the default active dataset."""
    if app_state["dataframe"] is not None and app_state["dataset_path"]:
        return app_state["dataframe"].copy()
    default_path = _default_dataset_path()
    if default_path is None:
        return pd.DataFrame()
    testing_path = str(Path(default_path).with_name("adult.test"))
    df, train_indices, test_indices = load_uci_adult_dataset(default_path, testing_path)
    _activate_dataset(
        df,
        dataset_key="uci_adult",
        name="UCI Adult Census Income",
        source="UCI Machine Learning Repository · adult.data + adult.test",
        path=default_path,
        target_column="income",
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return df.copy()


def _get_active_dataframe() -> pd.DataFrame:
    """Return the active DataFrame without triggering pandas truth-value errors."""
    df = app_state.get("dataframe")
    if df is None:
        return _ensure_default_dataset()
    return df.copy()


def _serialize_metric_row(metric_name: str, metric_value: float) -> Dict[str, Any]:
    """Create a normalized metric payload suitable for SQLite storage."""
    return {"metric_name": metric_name, "metric_value": float(metric_value)}


def _evaluation_response_summary(evaluation: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Return compact held-out metadata without serializing chart-sized curves."""
    if not evaluation:
        return None
    test_indices = evaluation.get("test_indices", [])
    digest = hashlib.sha256(",".join(str(index) for index in test_indices).encode("ascii")).hexdigest()
    return {
        "training_count": evaluation.get("training_count"),
        "testing_count": evaluation.get("testing_count"),
        "test_index_digest": digest,
    }


def _persist_experiment(
    name: str,
    status: str,
    method: str,
    dataset_path: str | None,
    result: Dict[str, Any],
    selected_indices: Iterable[int] = (),
    target_column: str | None = None,
    execution_time: float | None = None,
    training_records: int | None = None,
    evaluation_payload: Dict[str, Any] | None = None,
) -> None:
    """Persist experiment metadata into the database for history and reporting."""
    dataset_record = None
    if dataset_path:
        dataset_record = DatasetRecord.query.filter_by(file_path=dataset_path).first()
        if dataset_record is None and app_state.get("dataframe") is not None:
            active_frame = app_state["dataframe"]
            dataset_record = DatasetRecord(
                name=app_state.get("dataset_name") or Path(dataset_path).name,
                file_path=dataset_path,
                rows=int(len(active_frame)),
                columns=",".join(active_frame.columns.astype(str)),
                target_column=target_column or app_state.get("target_column"),
            )
            db.session.add(dataset_record)
            db.session.flush()
    experiment = ExperimentRecord(
        name=name,
        status=status,
        dataset_id=getattr(dataset_record, "id", None),
        model_name=result.get("model_name", "logistic_regression"),
        target_column=target_column or result.get("target_column"),
        task_type=result.get("task_type"),
        execution_time=execution_time,
        retraining_time=result.get("retraining_time"),
        evaluation_payload=json.dumps(evaluation_payload or result.get("original_evaluation") or {}),
    )
    db.session.add(experiment)
    db.session.flush()

    if result.get("original_metrics"):
        for key, value in result["original_metrics"].items():
            if value is not None and not pd.isna(value):
                db.session.add(EvaluationResultRecord(experiment_id=experiment.id, metric_name=f"original_{key}", metric_value=float(value)))

    result_metrics = result.get("unlearned_metrics", result.get("approximate_metrics", {}))
    if result_metrics:
        for key, value in result_metrics.items():
            if value is not None and not pd.isna(value):
                db.session.add(EvaluationResultRecord(experiment_id=experiment.id, metric_name=f"unlearned_{key}", metric_value=float(value)))

    selected_indices = [int(index) for index in selected_indices]
    db.session.add(UnlearningRequestRecord(
        experiment_id=experiment.id,
        forget_count=len(selected_indices),
        selected_indices=json.dumps(selected_indices),
        method=method,
        status=status,
        remaining_training_records=(
            max(0, training_records - len(selected_indices))
            if training_records is not None
            else None
        ),
    ))
    commit_started = time.perf_counter()
    db.session.commit()
    current_app.logger.info(
        "experiment persistence committed experiment_id=%s elapsed_seconds=%.3f",
        experiment.id,
        time.perf_counter() - commit_started,
    )


def _dashboard_evaluation_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the held-out chart data needed to rebuild the dashboard after refresh."""
    keys = (
        "class_labels",
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
        "feature_importance",
        "actual_vs_predicted",
        "residuals",
    )
    return {key: result.get(key) for key in keys}


def _saved_training_result(dataset_path: str | None, target_column: str) -> Dict[str, Any] | None:
    """Load the latest completed training metrics and chart data for this dataset target."""
    if not dataset_path:
        return None
    dataset_record = DatasetRecord.query.filter_by(file_path=dataset_path).first()
    if dataset_record is None:
        return None
    experiments = (
        ExperimentRecord.query
        .filter_by(
            dataset_id=dataset_record.id,
            target_column=target_column,
            status="completed",
        )
        .order_by(ExperimentRecord.created_at.desc())
        .all()
    )
    for experiment in experiments:
        if UnlearningRequestRecord.query.filter_by(experiment_id=experiment.id).first():
            continue
        metric_rows = EvaluationResultRecord.query.filter_by(experiment_id=experiment.id).all()
        metrics = {
            row.metric_name.removeprefix("original_"): float(row.metric_value)
            for row in metric_rows
            if row.metric_name.startswith("original_")
        }
        if not metrics:
            continue
        try:
            evaluation = json.loads(experiment.evaluation_payload or "{}")
        except (TypeError, json.JSONDecodeError):
            evaluation = {}
        return {
            "metrics": metrics,
            "model_name": experiment.model_name,
            "task_type": experiment.task_type,
            "training_time": experiment.execution_time,
            "original_evaluation": evaluation,
            "actual_vs_predicted": evaluation.get("actual_vs_predicted"),
            "residuals": evaluation.get("residuals"),
        }
    return None


def _saved_unlearning_history(
    dataset_path: str | None,
    target_column: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Read completed unlearning comparisons from SQLite for the active dataset target."""
    if not dataset_path:
        return []
    dataset_record = DatasetRecord.query.filter_by(file_path=dataset_path).first()
    if dataset_record is None:
        return []
    experiments = (
        ExperimentRecord.query
        .filter_by(
            dataset_id=dataset_record.id,
            target_column=target_column,
            status="completed",
        )
        .order_by(ExperimentRecord.created_at.desc())
        .all()
    )
    requests = {
        row.experiment_id: row
        for row in UnlearningRequestRecord.query.filter_by(status="completed").all()
        if row.experiment_id is not None
    }
    selected = [
        experiment for experiment in experiments
        if experiment.id in requests
    ][:limit]
    if not selected:
        return []
    metric_rows = EvaluationResultRecord.query.filter(
        EvaluationResultRecord.experiment_id.in_([item.id for item in selected])
    ).all()
    metrics_by_experiment: Dict[int, Dict[str, float]] = {}
    for row in metric_rows:
        metrics_by_experiment.setdefault(row.experiment_id, {})[row.metric_name] = float(row.metric_value)

    history = []
    for experiment in selected:
        request_record = requests[experiment.id]
        saved_metrics = metrics_by_experiment.get(experiment.id, {})
        try:
            evaluation = json.loads(experiment.evaluation_payload or "{}")
        except (TypeError, json.JSONDecodeError):
            evaluation = {}
        history.append({
            "id": experiment.id,
            "name": experiment.name,
            "model_name": experiment.model_name,
            "target_column": experiment.target_column,
            "task_type": experiment.task_type,
            "method": request_record.method,
            "forgotten_records": request_record.forget_count,
            "remaining_training_records": request_record.remaining_training_records,
            "execution_time": experiment.execution_time,
            "retraining_time": experiment.retraining_time,
            "created_at": experiment.created_at.isoformat(),
            "original_metrics": {
                key.removeprefix("original_"): value
                for key, value in saved_metrics.items()
                if key.startswith("original_")
            },
            "post_metrics": {
                key.removeprefix("unlearned_"): value
                for key, value in saved_metrics.items()
                if key.startswith("unlearned_")
            },
            "evaluation": evaluation,
        })
    return history


def _upsert_dataset_record(name: str, path: str, df: pd.DataFrame, target_column: str) -> None:
    """Keep the active dataset's database metadata aligned with the selected source."""
    record = DatasetRecord.query.filter_by(file_path=path).first()
    if record is None:
        record = DatasetRecord(name=name, file_path=path)
        db.session.add(record)
    record.name = name
    record.rows = int(len(df))
    record.columns = ",".join(df.columns.astype(str))
    record.target_column = target_column
    db.session.commit()


def _build_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a dataset summary payload for API responses and the dashboard."""
    if df.empty:
        return {"rows": 0, "columns": [], "shape": [0, 0], "missing_values": {} }
    target = app_state.get("target_column") or detect_target_column(df, "income")
    split = app_state.get("split_indices") or {}
    try:
        task_type = detect_target_task(df[target])
        supported_models = get_supported_models(task_type)
    except (KeyError, ValueError):
        task_type = "unsupported"
        supported_models = []
    class_distribution = df[target].value_counts().to_dict() if task_type.endswith("classification") else {}
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "target_column": target,
        "task_type": task_type,
        "supported_models": supported_models,
        "dataset_key": app_state.get("dataset_key"),
        "name": app_state.get("dataset_name") or Path(app_state.get("dataset_path") or "Dataset").name,
        "source": app_state.get("dataset_source") or "Uploaded CSV",
        "training_rows": len(split.get("train", [])),
        "testing_rows": len(split.get("test", [])),
        "missing_values": df.isna().sum().to_dict(),
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
    }


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values into JSON-serializable Python scalar types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _build_dashboard_payload() -> Dict[str, Any]:
    """Assemble the real dashboard metrics and chart payload for the frontend."""
    df = _get_active_dataframe()
    target_column = app_state.get("target_column", "income")
    original_result = app_state.get("original_result")
    unlearned_result = app_state.get("unlearned_result")
    saved_unlearning = _saved_unlearning_history(app_state.get("dataset_path"), target_column)
    if original_result is None:
        original_result = _saved_training_result(app_state.get("dataset_path"), target_column)
    if original_result is None and saved_unlearning:
        latest_saved = saved_unlearning[0]
        if latest_saved["original_metrics"]:
            original_result = {
                "metrics": latest_saved["original_metrics"],
                "model_name": latest_saved["model_name"],
                "task_type": latest_saved["task_type"],
                "original_evaluation": latest_saved["evaluation"],
            }
    if unlearned_result is None and saved_unlearning:
        latest_saved = saved_unlearning[0]
        unlearned_result = {
            "unlearned_metrics": latest_saved["post_metrics"],
            "method": latest_saved["method"],
        }
    original_metrics = original_result.get("metrics", {}) if original_result else {}
    original_evaluation = original_result.get("original_evaluation", original_result) if original_result else {}
    unlearned_metrics = {}
    if unlearned_result:
        unlearned_metrics = unlearned_result.get(
            "unlearned_metrics", unlearned_result.get("approximate_metrics", {}) 
        )
    task_type = (
        original_result.get("task_type")
        if original_result
        else detect_target_task(df[target_column])
    )

    if df.empty:
        return {
            "stats": {
                "total_records": 0,
                "training_records": None,
                "testing_records": None,
                "task_type": None,
                "forgotten_records": 0,
                "retained_records": 0,
                "remaining_training_records": 0,
                "original_model_accuracy": None,
                "unlearned_model_accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "training_time": None,
                "unlearning_time": None,
                "retraining_time": None,
            },
            "charts": {},
            "has_model": False,
            "has_unlearning": False,
            "task_type": None,
            "unlearning_history": [],
            "history": [],
        }

    total_records = len(df)
    active_split = _active_split()
    training_records = len(original_result["X_train"]) if original_result and "X_train" in original_result else len(active_split["train"])
    testing_records = len(original_result["X_test"]) if original_result and "X_test" in original_result else len(active_split["test"])
    metric_names = ["mae", "mse", "rmse", "r2"] if task_type == "regression" else ["accuracy", "precision", "recall", "f1"]
    target_counts = (
        df[target_column].value_counts()
        if target_column in df.columns and task_type.endswith("classification")
        else pd.Series(dtype="int64")
    )
    if task_type == "regression" and target_column in df.columns:
        target_values = pd.to_numeric(df[target_column], errors="coerce").dropna()
        bin_count = min(20, max(5, int(len(target_values) ** 0.5)))
        histogram, bin_edges = np.histogram(target_values, bins=bin_count)
        distribution_labels = [f"{bin_edges[index]:g}–{bin_edges[index + 1]:g}" for index in range(len(histogram))]
        distribution_values = histogram
    else:
        distribution_labels = [str(label) for label in target_counts.index.tolist()]
        distribution_values = target_counts.values
    numeric_features = df.drop(columns=[target_column], errors="ignore").select_dtypes(include="number")
    correlation = numeric_features.corr().fillna(0) if numeric_features.shape[1] else pd.DataFrame()
    missing_counts = df.isna().sum().sort_values(ascending=False)
    unlearning_results = app_state.get("unlearning_results", {})
    exact_result = unlearning_results.get("exact")
    approximate_result = unlearning_results.get("approximate")
    saved_by_method = {}
    for saved_run in saved_unlearning:
        saved_by_method.setdefault(saved_run["method"], saved_run)
    exact_metrics = (
        exact_result.get("unlearned_metrics")
        if exact_result
        else saved_by_method.get("exact", {}).get("post_metrics")
    )
    approximate_metrics = (
        approximate_result.get("approximate_metrics")
        if approximate_result
        else saved_by_method.get("approximate", {}).get("post_metrics")
    )
    comparison_values = lambda metrics: [metrics.get(key) for key in metric_names] if metrics else None
    primary_metric = "r2" if task_type == "regression" else "accuracy"
    latest_saved_unlearning = saved_unlearning[0] if saved_unlearning else {}
    unlearning_original_metrics = (
        unlearned_result.get("original_metrics")
        if unlearned_result and unlearned_result.get("original_metrics")
        else latest_saved_unlearning.get("original_metrics") or original_metrics
    )
    unlearning_model_name = (
        unlearned_result.get("model_name")
        if unlearned_result
        else latest_saved_unlearning.get("model_name")
    )
    method_metrics = {
        "Original": unlearning_original_metrics or None,
        "Exact retraining": exact_metrics,
        "Approximate": approximate_metrics,
    }
    has_unlearning = bool(unlearned_result)
    forgotten_records = (
        len(app_state.get("forget_indices", []))
        if app_state.get("unlearned_result")
        else latest_saved_unlearning.get("forgotten_records", 0)
    )
    remaining_training_records = (
        max(0, training_records - forgotten_records)
        if has_unlearning
        else training_records
    )
    retained_records = remaining_training_records
    unlearning_method = (
        app_state.get("unlearning_method")
        or latest_saved_unlearning.get("method")
    )
    training_time = (
        app_state.get("training_time")
        if app_state.get("original_result")
        else (original_result or {}).get("training_time")
    )
    unlearning_time = (
        app_state.get("unlearning_time")
        if app_state.get("unlearned_result")
        else latest_saved_unlearning.get("execution_time")
    )
    retraining_time = (
        (unlearning_results.get("exact") or {}).get("retraining_time")
        or latest_saved_unlearning.get("retraining_time")
    )

    payload = {
        "stats": {
            "total_records": total_records,
            "training_records": training_records,
            "testing_records": testing_records,
            "forgotten_records": forgotten_records,
            "retained_records": retained_records,
            "original_model_accuracy": original_metrics.get("accuracy"),
            "unlearned_model_accuracy": unlearned_metrics.get("accuracy"),
            "task_type": task_type,
            "metrics": original_metrics,
            "unlearning_original_metrics": unlearning_original_metrics,
            "unlearned_metrics": unlearned_metrics,
            "training_time": training_time,
            "unlearning_time": unlearning_time,
            "retraining_time": retraining_time,
            "model_name": original_result.get("model_name") if original_result else None,
            "unlearning_model_name": unlearning_model_name,
            "target_column": target_column,
            "unlearning_method": unlearning_method,
            "remaining_training_records": remaining_training_records,
        },
        "charts": {
            "performance": {
                "labels": ["Original", "Unlearned"],
                "series": [
                    unlearning_original_metrics.get(primary_metric),
                    unlearned_metrics.get(primary_metric),
                ],
            },
            "class_distribution": {
                "labels": distribution_labels,
                "values": [int(value) for value in distribution_values],
            },
            "comparison_metrics": {
                "labels": [key.replace("_", " ").upper() for key in metric_names],
                "original": comparison_values(unlearning_original_metrics),
                "unlearned": comparison_values(unlearned_metrics),
            },
            "method_comparison": {
                method: comparison_values(metrics)
                for method, metrics in method_metrics.items()
            },
            "actual_vs_predicted": original_result.get("actual_vs_predicted") if original_result else None,
            "residuals": original_result.get("residuals") if original_result else None,
            "confusion_matrix": {
                "labels": original_evaluation.get("class_labels", []),
                "values": original_evaluation.get("confusion_matrix", []),
            },
            "roc": original_evaluation.get("roc_curve"),
            "auc": original_metrics.get("roc_auc", original_metrics.get("auc")) if original_metrics else None,
            "precision_recall": original_evaluation.get("precision_recall_curve"),
            "actual_vs_predicted": original_result.get("actual_vs_predicted") if original_result else None,
            "residuals": original_result.get("residuals") if original_result else None,
            "feature_importance": original_evaluation.get("feature_importance", []),
            "missing_values": {
                "labels": [str(column) for column in missing_counts.index[:15]],
                "values": [int(value) for value in missing_counts.values[:15]],
            },
            "correlation": {
                "labels": [str(column) for column in correlation.columns],
                "values": [[float(value) for value in row] for row in correlation.to_numpy().tolist()],
            },
            "forgetting_records": {
                "labels": ["Forgotten", "Retained"],
                "values": [forgotten_records, remaining_training_records],
            },
            "timings": {
                "labels": ["Training", "Unlearning"],
                "values": [training_time, unlearning_time],
            },
        },
        "dataset": _build_dataset_summary(df),
        "task_type": task_type,
        "has_model": bool(original_result),
        "has_unlearning": has_unlearning,
        "unlearning_history": saved_unlearning,
        "history": [
            {
                "name": item.name,
                "status": item.status,
                "model_name": item.model_name,
                "created_at": item.created_at.isoformat(),
            }
            for item in ExperimentRecord.query.order_by(ExperimentRecord.created_at.desc()).limit(10).all()
        ],
    }
    return payload


def create_app() -> Flask:
    """Construct the Flask app and register all working routes."""
    app = Flask(__name__)
    app.config.from_object(CONFIG)
    app.logger.setLevel(logging.INFO)
    os.makedirs(CONFIG.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(CONFIG.PROCESSED_DATASET_FOLDER, exist_ok=True)
    os.makedirs(CONFIG.MODEL_ORIGINAL_FOLDER, exist_ok=True)
    os.makedirs(CONFIG.MODEL_UNLEARNED_FOLDER, exist_ok=True)
    os.makedirs(CONFIG.RESULTS_FOLDER, exist_ok=True)
    os.makedirs(CONFIG.GRAPHS_FOLDER, exist_ok=True)
    os.makedirs(CONFIG.REPORTS_FOLDER, exist_ok=True)
    init_db(app)

    @app.context_processor
    def inject_active_navigation() -> Dict[str, str]:
        """Keep the shared sidebar selection aligned with the current page."""
        active_page = {
            "/": "dashboard",
            "/datasets": "datasets",
            "/training": "models",
            "/model-explorer": "models",
            "/unlearning": "unlearning",
            "/evaluation": "evaluation",
            "/history": "history",
            "/reports": "reports",
            "/documentation": "documentation",
            "/settings": "documentation",
        }.get(request.path, "")
        return {"active_page": active_page}

    @app.route('/')
    def index() -> str:
        """Render the main dashboard."""
        return render_template('dashboard.html')

    @app.route('/datasets')
    def datasets_page() -> str:
        """Render the dataset explorer page."""
        return render_template('datasets.html')

    @app.route('/training')
    def training_page() -> str:
        """Render the model training page."""
        return render_template('training.html')

    @app.route('/model-explorer')
    def model_explorer_page() -> str:
        """Render the model explorer page."""
        return render_template('model_explorer.html')

    @app.route('/unlearning')
    def unlearning_page() -> str:
        """Render the unlearning analytics page."""
        return render_template('unlearning.html')

    @app.route('/evaluation')
    def evaluation_page() -> str:
        """Render the evaluation center."""
        return render_template('evaluation.html')

    @app.route('/history')
    def history_page() -> str:
        """Render the experiment history view."""
        return render_template('history.html')

    @app.route('/reports')
    def reports_page() -> str:
        """Render report export controls."""
        return render_template('reports.html')

    @app.route('/documentation')
    def docs_page() -> str:
        """Render the documentation page."""
        return render_template('documentation.html')

    @app.route('/settings')
    def settings_page() -> str:
        """Render the settings page."""
        return render_template('documentation.html')

    @app.route('/health')
    def health() -> Any:
        """Return a lightweight health-check payload for startup verification."""
        return jsonify({"status": "ok"})

    @app.route('/api/dashboard')
    def api_dashboard() -> Any:
        """Provide the real KPI values and chart data for the dashboard."""
        payload = _build_dashboard_payload()
        if request.args.get("view") == "unlearning":
            payload["charts"] = {
                "method_comparison": payload.get("charts", {}).get("method_comparison", {}),
                "comparison_metrics": payload.get("charts", {}).get("comparison_metrics", {}),
                "forgetting_records": payload.get("charts", {}).get("forgetting_records", {}),
                "timings": payload.get("charts", {}).get("timings", {}),
            }
        return jsonify(payload)

    @app.route('/api/dataset')
    def api_dataset() -> Any:
        """Return the active dataset summary and target column metadata."""
        df = _get_active_dataframe()
        return jsonify(_build_dataset_summary(df))

    @app.route('/api/datasets')
    def api_dataset_catalog() -> Any:
        """List the official UCI dataset and existing uploaded CSVs."""
        _ensure_default_dataset()
        return jsonify({
            "active_dataset_key": app_state.get("dataset_key"),
            "datasets": _available_datasets(),
        })

    @app.route('/api/dataset/activate', methods=['POST'])
    def api_activate_dataset() -> Any:
        """Activate a known UCI or uploaded dataset and establish its fixed split."""
        payload = request.get_json(silent=True) or {}
        dataset_key = payload.get("dataset_key")
        if dataset_key == "uci_adult":
            training_path = _default_dataset_path()
            if not training_path:
                return jsonify({"error": "The official adult.data and adult.test files are required."}), 404
            testing_path = str(Path(training_path).with_name("adult.test"))
            try:
                df, train_indices, test_indices = load_uci_adult_dataset(training_path, testing_path)
                _activate_dataset(
                    df,
                    dataset_key="uci_adult",
                    name="UCI Adult Census Income",
                    source="UCI Machine Learning Repository · adult.data + adult.test",
                    path=training_path,
                    target_column="income",
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            except (OSError, ValueError) as error:
                return jsonify({"error": f"Unable to load the official UCI Adult files: {error}"}), 400
        elif isinstance(dataset_key, str) and dataset_key.startswith("upload:"):
            filename = sanitize_filename(dataset_key.removeprefix("upload:"))
            destination = (Path(CONFIG.UPLOAD_FOLDER) / filename).resolve()
            if destination.parent != Path(CONFIG.UPLOAD_FOLDER).resolve() or not validate_csv_file(destination):
                return jsonify({"error": "The selected uploaded CSV is no longer available."}), 404
            try:
                df = load_dataset(destination)
                target = detect_target_column(df, "income")
                if target is None:
                    return jsonify({"error": "The selected dataset has no target column."}), 400
                _activate_dataset(
                    df,
                    dataset_key=dataset_key,
                    name=filename,
                    source="Uploaded CSV",
                    path=str(destination),
                    target_column=target,
                )
            except (OSError, ValueError) as error:
                return jsonify({"error": f"Unable to activate the selected CSV: {error}"}), 400
        else:
            return jsonify({"error": "Select a dataset from the available dataset list."}), 400

        _upsert_dataset_record(
            app_state["dataset_name"],
            app_state["dataset_path"],
            app_state["dataframe"],
            app_state["target_column"],
        )
        return jsonify({"status": "ok", "dataset": _build_dataset_summary(app_state["dataframe"])})

    @app.route('/api/dataset/upload', methods=['POST'])
    def api_upload_dataset() -> Any:
        """Upload a CSV or data file, validate it, and make it the active dataset."""
        uploaded = request.files.get('file')
        if uploaded is None or uploaded.filename == '':
            return jsonify({"error": "No file uploaded."}), 400
        safe_name = sanitize_filename(uploaded.filename)
        destination = Path(CONFIG.UPLOAD_FOLDER) / safe_name
        uploaded.save(str(destination))

        if not validate_csv_file(destination):
            return jsonify({"error": "Uploaded file could not be validated as tabular data."}), 400

        try:
            df = load_dataset(destination)
            target = detect_target_column(df, "income")
            if target is None:
                return jsonify({"error": "The uploaded CSV has no target column."}), 400
            _activate_dataset(
                df,
                dataset_key=f"upload:{safe_name}",
                name=safe_name,
                source="Uploaded CSV",
                path=str(destination),
                target_column=target,
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        _upsert_dataset_record(safe_name, str(destination), df, target)
        return jsonify({"status": "ok", "dataset": _build_dataset_summary(df)})

    @app.route('/api/dataset/preview')
    def api_dataset_preview() -> Any:
        """Return a small preview of the active dataset for the explorer UI."""
        df = _get_active_dataframe()
        preview = df.head(20).fillna("NA").to_dict(orient="records")
        return jsonify({"rows": preview, "columns": df.columns.tolist(), "shape": [int(df.shape[0]), int(df.shape[1])]})

    @app.route('/api/target', methods=['POST'])
    def api_target_column() -> Any:
        """Set the active target column and return dataset metadata."""
        payload = request.get_json(silent=True) or {}
        target_column = payload.get('target_column')
        df = _get_active_dataframe()
        if target_column not in df.columns:
            return jsonify({"error": "Target column not found in the active dataset."}), 400
        try:
            task_type = detect_target_task(df[target_column])
        except ValueError as error:
            return jsonify({"error": f"Target '{target_column}' is unsupported: {error}"}), 400
        if target_column != app_state.get("target_column"):
            df = _get_active_dataframe()
            split = app_state.get("partition_indices")
            try:
                _activate_dataset(
                    df,
                    dataset_key=app_state["dataset_key"],
                    name=app_state["dataset_name"],
                    source=app_state["dataset_source"],
                    path=app_state["dataset_path"],
                    target_column=target_column,
                    train_indices=split["train"] if split else None,
                    test_indices=split["test"] if split else None,
                )
            except ValueError as error:
                return jsonify({"error": f"Target '{target_column}' is unsupported: {error}"}), 400
        dataset_path = app_state.get("dataset_path")
        if dataset_path:
            _upsert_dataset_record(app_state["dataset_name"], dataset_path, df, target_column)
        return jsonify({
            "status": "ok",
            "target_column": target_column,
            "task_type": task_type,
            "supported_models": get_supported_models(task_type),
        })

    @app.route('/api/unlearning/selection')
    def api_unlearning_selection() -> Any:
        """Return a page of selectable training records, never held-out rows."""
        df = _get_active_dataframe()
        if df.empty:
            return jsonify({"error": "No active dataset is available."}), 400
        try:
            offset = int(request.args.get("offset", 0))
            limit = int(request.args.get("limit", 10))
        except ValueError:
            return jsonify({"error": "Page offset and size must be integers."}), 400
        if offset < 0 or limit < 1 or limit > 50:
            return jsonify({"error": "Page offset must be non-negative and page size must be from 1 to 50."}), 400

        target_column = app_state.get("target_column")
        try:
            split = _active_split()
            X_train, X_test, _, _ = split_training_data(
                df, target_column, train_indices=split["train"], test_indices=split["test"]
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        # The selection table is sourced from the same deterministic split as training.
        training_indices = [int(index) for index in X_train.index]
        page_indices = training_indices[offset:offset + limit]
        rows = []
        for index in page_indices:
            row = {"index": index}
            for column, value in df.loc[index].items():
                row[str(column)] = "NA" if pd.isna(value) else _to_python_scalar(value)
            rows.append(row)
        return jsonify({
            "target_column": target_column,
            "training_count": len(training_indices),
            "testing_count": len(X_test),
            "offset": offset,
            "limit": limit,
            "rows": rows,
        })

    @app.route('/api/train', methods=['POST'])
    def api_train_model() -> Any:
        """Train the selected model on the active dataset, persist it, and return metrics."""
        payload = request.get_json(silent=True) or {}
        model_name = payload.get("model_name", "logistic_regression")
        target_column = payload.get("target_column", app_state.get("target_column"))
        df = _get_active_dataframe()
        if df.empty:
            return jsonify({"error": "No dataset is selected for training."}), 400
        if target_column != app_state.get("target_column"):
            return jsonify({"error": "Choose this target in Dataset Explorer before training."}), 409

        start = time.perf_counter()
        split = _active_split()
        try:
            model_result = train_model(
                model_name,
                df,
                target_column,
                train_indices=split["train"],
                test_indices=split["test"],
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        elapsed = time.perf_counter() - start
        app_state["original_result"] = model_result
        app_state["training_time"] = elapsed
        app_state["target_column"] = target_column
        app_state["unlearned_result"] = None
        app_state["unlearning_results"] = {}
        app_state["forget_indices"] = []
        app_state["unlearning_method"] = None
        app_state["unlearning_time"] = 0.0

        model_path = Path(CONFIG.MODEL_ORIGINAL_FOLDER) / f"{model_name}_{int(time.time())}.joblib"
        save_model(model_result["model"], str(model_path))

        history = ExperimentRecord(
            name=f"{model_name} training",
            status="completed",
            model_name=model_name,
            target_column=target_column,
            task_type=model_result["task_type"],
            execution_time=elapsed,
            evaluation_payload=json.dumps(_dashboard_evaluation_payload(model_result)),
        )
        active_path = app_state.get("dataset_path")
        dataset_record = DatasetRecord.query.filter_by(file_path=active_path).first() if active_path else None
        if active_path and dataset_record is None:
            dataset_record = DatasetRecord(
                name=Path(active_path).name,
                file_path=active_path,
                rows=int(len(df)),
                columns=",".join(df.columns.astype(str)),
                target_column=target_column,
            )
            db.session.add(dataset_record)
            db.session.flush()
        history.dataset_id = dataset_record.id if dataset_record else None
        db.session.add(history)
        db.session.flush()
        for metric_name, metric_value in model_result["metrics"].items():
            if metric_value is not None and not pd.isna(metric_value):
                db.session.add(EvaluationResultRecord(
                    experiment_id=history.id,
                    metric_name=f"original_{metric_name}",
                    metric_value=float(metric_value),
                ))
        db.session.add(ModelRecord(experiment_id=history.id, model_type="original", file_path=str(model_path)))
        db.session.commit()

        payload_response = {
            "status": "ok",
            "metrics": model_result["metrics"],
            "task_type": model_result["task_type"],
            "model_name": model_name,
            "training_time": elapsed,
            "confusion_matrix": model_result.get("confusion_matrix", []),
            "feature_importance": model_result.get("feature_importance", []),
            "actual_vs_predicted": model_result.get("actual_vs_predicted"),
            "residuals": model_result.get("residuals"),
        }
        return jsonify(payload_response)

    @app.route('/api/unlearn', methods=['POST'])
    def api_unlearn() -> Any:
        """Run the selected exact or approximate unlearning method on the active dataset."""
        payload = request.get_json(silent=True) or {}
        method = payload.get("method", "exact")
        forget_indices = payload.get("forget_indices", [])
        target_column = payload.get("target_column", app_state.get("target_column"))
        model_name = payload.get("model_name", "logistic_regression")
        df = _get_active_dataframe()

        if df.empty:
            return jsonify({"error": "No dataset is selected for unlearning."}), 400
        if method not in {"exact", "approximate"}:
            return jsonify({"error": "Choose exact or approximate unlearning."}), 400
        if target_column != app_state.get("target_column"):
            return jsonify({"error": "The target changed. Reload the page and use the active dataset target."}), 409
        if not isinstance(forget_indices, list):
            return jsonify({"error": "Select one or more training records."}), 400
        if any(isinstance(index, bool) or not isinstance(index, int) for index in forget_indices):
            return jsonify({"error": "Selected record indices must be integers."}), 400
        if len(set(forget_indices)) != len(forget_indices):
            return jsonify({"error": "A record can only be selected once."}), 400
        forget_count = payload.get("forget_count", len(forget_indices))
        if isinstance(forget_count, bool) or not isinstance(forget_count, int) or forget_count <= 0:
            return jsonify({"error": "Forget count must be a whole number greater than zero."}), 400
        if forget_count != len(forget_indices):
            return jsonify({"error": "Forget count must match the selected training records."}), 400

        try:
            split = _active_split()
            X_train, _, _, _ = split_training_data(
                df, target_column, train_indices=split["train"], test_indices=split["test"]
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        training_indices = set(int(index) for index in X_train.index)
        if forget_count > len(training_indices):
            return jsonify({"error": f"You can select at most {len(training_indices)} training records."}), 400
        if any(index not in training_indices for index in forget_indices):
            return jsonify({"error": "Held-out or unknown records cannot be selected for forgetting."}), 400
        if forget_count == len(training_indices):
            return jsonify({"error": "At least one training record must remain after forgetting."}), 400

        request_id = uuid.uuid4().hex[:10]
        request_started = time.perf_counter()
        current_app.logger.info(
            "unlearning accepted request_id=%s method=%s dataset=%s rows=%s target=%s selected=%s train=%s test=%s",
            request_id,
            method,
            app_state.get("dataset_name"),
            len(df),
            target_column,
            len(forget_indices),
            len(split["train"]),
            len(split["test"]),
        )
        stage = "model computation"
        model_started = time.perf_counter()
        try:
            if method == "approximate":
                result = approximate_unlearning_experiment(
                    df, target_column, forget_indices, model_name=model_name,
                    train_indices=split["train"], test_indices=split["test"],
                )
            else:
                result = exact_retraining_unlearning(
                    df, target_column, forget_indices, model_name=model_name,
                    train_indices=split["train"], test_indices=split["test"],
                )
        except ValueError as error:
            current_app.logger.warning(
                "unlearning rejected during %s request_id=%s error=%s",
                stage,
                request_id,
                error,
            )
            return jsonify({"error": str(error)}), 400
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "unlearning failed during %s request_id=%s",
                stage,
                request_id,
            )
            return jsonify({
                "error": f"Unlearning failed during {method} model computation.",
                "request_id": request_id,
            }), 500
        model_elapsed = time.perf_counter() - model_started
        current_app.logger.info(
            "unlearning model computation complete request_id=%s method=%s elapsed_seconds=%.3f",
            request_id,
            method,
            model_elapsed,
        )

        stage = "database persistence"
        persistence_started = time.perf_counter()
        current_app.logger.info(
            "unlearning persistence started request_id=%s method=%s",
            request_id,
            method,
        )
        try:
            model_path = Path(CONFIG.MODEL_UNLEARNED_FOLDER) / f"{method}_{int(time.time())}.joblib"
            if "model" in result:
                save_model(result["model"], str(model_path))
            _persist_experiment(
                name=f"{method} unlearning",
                status="completed",
                method=method,
                dataset_path=app_state.get("dataset_path"),
                result=result,
                selected_indices=forget_indices,
                target_column=target_column,
                execution_time=time.perf_counter() - request_started,
                training_records=len(training_indices),
            )
        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "unlearning failed during %s request_id=%s",
                stage,
                request_id,
            )
            return jsonify({
                "error": "Unlearning results could not be saved. No success was reported.",
                "request_id": request_id,
            }), 500
        persistence_elapsed = time.perf_counter() - persistence_started
        elapsed = time.perf_counter() - request_started

        if not app_state.get("original_result") and result.get("original_metrics"):
            app_state["original_result"] = {
                "metrics": result["original_metrics"],
                "model_name": model_name,
                "task_type": result.get("task_type"),
                "original_evaluation": result.get("original_evaluation", {}),
            }
        app_state["unlearned_result"] = result
        app_state.setdefault("unlearning_results", {})[method] = result
        app_state["forget_indices"] = list(forget_indices)
        app_state["unlearning_time"] = elapsed
        app_state["unlearning_method"] = method

        response_result = {
            "method": result.get("method"),
            "original_metrics": result.get("original_metrics", {}),
            "unlearned_metrics": result.get("unlearned_metrics"),
            "approximate_metrics": result.get("approximate_metrics"),
            "comparison": result.get("comparison", {}),
            "notes": result.get("notes", []),
            "original_evaluation": _evaluation_response_summary(result.get("original_evaluation")),
            "unlearned_evaluation": _evaluation_response_summary(result.get("unlearned_evaluation")),
        }
        response_payload = {
            "status": "ok",
            "request_id": request_id,
            "method": method,
            "task_type": result.get("task_type"),
            "unlearning_time": elapsed,
            "result": response_result,
            "comparison": result.get("comparison", {}) if method == "exact" else result.get("approximate_metrics", {}),
        }
        current_app.logger.info(
            "unlearning completed request_id=%s method=%s model_seconds=%.3f persistence_seconds=%.3f total_seconds=%.3f response_bytes=%s",
            request_id,
            method,
            model_elapsed,
            persistence_elapsed,
            time.perf_counter() - request_started,
            len(json.dumps(response_payload, separators=(",", ":"))),
        )
        return jsonify(response_payload)

    @app.route('/api/history')
    def api_history() -> Any:
        """Return the recent experiment history as JSON."""
        records = ExperimentRecord.query.order_by(ExperimentRecord.created_at.desc()).all()
        unlearning_requests = {
            item.experiment_id: item
            for item in UnlearningRequestRecord.query.order_by(UnlearningRequestRecord.created_at.desc()).all()
            if item.experiment_id is not None
        }
        history = []
        for item in records:
            unlearning = unlearning_requests.get(item.id)
            try:
                selected_indices = json.loads(unlearning.selected_indices or "[]") if unlearning else []
            except (TypeError, json.JSONDecodeError):
                selected_indices = []
            history.append({
                "id": item.id,
                "name": item.name,
                "status": item.status,
                "model_name": item.model_name,
                "target_column": item.target_column,
                "task_type": item.task_type,
                "execution_time": item.execution_time,
                "retraining_time": item.retraining_time,
                "created_at": item.created_at.isoformat(),
                "method": unlearning.method if unlearning else None,
                "forget_count": unlearning.forget_count if unlearning else 0,
                "remaining_training_records": unlearning.remaining_training_records if unlearning else None,
                "selected_indices": selected_indices,
            })
        return jsonify({"history": history})

    @app.route('/api/report/export/<report_type>')
    def api_export_report(report_type: str) -> Any:
        """Export the most recent experiment results as CSV or JSON."""
        original_result = app_state.get("original_result")
        if not original_result:
            return jsonify({"error": "Train a model before exporting a report."}), 409
        unlearned_result = app_state.get("unlearned_result") or {}
        summary = generate_experiment_summary({
            "method": "latest",
            "original_metrics": original_result.get("metrics", {}),
            "unlearned_metrics": unlearned_result.get("unlearned_metrics", {}) or unlearned_result.get("approximate_metrics", {}),
        })

        if report_type == "json":
            export_path = Path(CONFIG.REPORTS_FOLDER) / "latest_report.json"
            write_json_report(str(export_path), summary)
            return send_file(export_path, mimetype="application/json", as_attachment=True, download_name="latest_report.json")

        export_path = Path(CONFIG.REPORTS_FOLDER) / "latest_report.csv"
        pd.DataFrame([summary]).to_csv(export_path, index=False)
        return send_file(export_path, mimetype="text/csv", as_attachment=True, download_name="latest_report.csv")

    @app.route('/api/model-explorer')
    def api_model_explorer() -> Any:
        """Provide feature importance and comparison data for the model explorer page."""
        result = app_state.get("original_result") or _saved_training_result(
            app_state.get("dataset_path"),
            app_state.get("target_column", "income"),
        )
        if not result:
            return jsonify({"status": "empty", "feature_importance": [], "accuracy": None, "metrics": {}})
        return jsonify({
            "status": "ok",
            "task_type": result.get("task_type", result.get("metrics", {}).get("task_type")),
            "feature_importance": result.get("feature_importance", result.get("original_evaluation", {}).get("feature_importance", [])),
            "accuracy": result.get("metrics", {}).get("accuracy"),
            "model_name": result.get("model_name", "logistic_regression"),
            "metrics": result.get("metrics", {}),
            "actual_vs_predicted": result.get("actual_vs_predicted"),
            "residuals": result.get("residuals"),
            "confusion_matrix": result.get("confusion_matrix", result.get("original_evaluation", {}).get("confusion_matrix", [])),
            "class_labels": result.get("class_labels", result.get("original_evaluation", {}).get("class_labels", [])),
        })

    @app.route('/api/summary')
    def summary() -> Any:
        """Legacy summary endpoint for compatibility with the dashboard and tests."""
        payload = _build_dashboard_payload()
        return jsonify({
            "experiments": len(ExperimentRecord.query.all()),
            "datasets": len(DatasetRecord.query.all()),
            "models": len(ModelRecord.query.all()),
            "unlearning_tasks": len(UnlearningRequestRecord.query.all()),
            "status": "ok",
            **payload.get("stats", {}),
        })

    @app.route('/datasets.html')
    def legacy_datasets() -> str:
        """Compatibility alias for the dataset page."""
        return render_template('datasets.html')

    active_df = _ensure_default_dataset()
    if not active_df.empty and app_state.get("dataset_path"):
        with app.app_context():
            _upsert_dataset_record(
                app_state["dataset_name"] or Path(app_state["dataset_path"]).name,
                app_state["dataset_path"],
                active_df,
                app_state["target_column"],
            )
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
