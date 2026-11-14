"""SQLAlchemy models used by ForgetLab for experiment tracking.

Each table stores a different piece of metadata: the dataset, trained models,
requested unlearning tasks, and evaluation scores for later analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class DatasetRecord(db.Model):
    """Metadata row for a user-uploaded or imported dataset."""

    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    rows = db.Column(db.Integer, default=0)
    columns = db.Column(db.Text, nullable=True)
    target_column = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class ExperimentRecord(db.Model):
    """Represents a full training or unlearning experiment."""

    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), default="created")
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=True)
    model_name = db.Column(db.String(100), default="logistic_regression")
    target_column = db.Column(db.String(100), nullable=True)
    task_type = db.Column(db.String(40), nullable=True)
    execution_time = db.Column(db.Float, nullable=True)
    retraining_time = db.Column(db.Float, nullable=True)
    evaluation_payload = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ModelRecord(db.Model):
    """Stores filesystem path information for stored original or unlearned models."""

    __tablename__ = "models"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=True)
    model_type = db.Column(db.String(50), default="original")
    file_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class UnlearningRequestRecord(db.Model):
    """Tracks a requested forgetting operation and its details."""

    __tablename__ = "unlearning_requests"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=True)
    forget_count = db.Column(db.Integer, default=0)
    selected_indices = db.Column(db.Text, nullable=True)
    method = db.Column(db.String(80), default="exact_retraining")
    status = db.Column(db.String(50), default="pending")
    remaining_training_records = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class EvaluationResultRecord(db.Model):
    """Stores sample-level metrics and summary values for one evaluation run."""

    __tablename__ = "evaluation_results"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=True)
    metric_name = db.Column(db.String(100), nullable=False)
    metric_value = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
