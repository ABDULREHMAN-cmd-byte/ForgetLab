"""Database initialization utilities for the ForgetLab application.

The app uses SQLite and SQLAlchemy to record experiment metadata, dataset
information, model locations, and evaluation results in a persistent store.
"""

from __future__ import annotations

from flask import Flask
from sqlalchemy import inspect, text

from config import CONFIG
from models_db import db


def init_db(app: Flask) -> None:
    """Bind the Flask app to SQLAlchemy and create all tables.

    Args:
        app: The Flask application instance.

    Returns:
        None
    """
    app.config["SQLALCHEMY_DATABASE_URI"] = CONFIG.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = CONFIG.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)
    with app.app_context():
        db.create_all()
        schema_additions = {
            "experiments": {
                "target_column": "VARCHAR(100)",
                "task_type": "VARCHAR(40)",
                "execution_time": "FLOAT",
                "retraining_time": "FLOAT",
                "evaluation_payload": "TEXT",
            },
            "unlearning_requests": {
                "selected_indices": "TEXT",
                "remaining_training_records": "INTEGER",
            },
        }
        for table_name, columns in schema_additions.items():
            existing_columns = {
                column["name"] for column in inspect(db.engine).get_columns(table_name)
            }
            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    db.session.execute(
                        text(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN {column_name} {column_type}"
                        )
                    )
        db.session.execute(text("SELECT 1"))
        db.session.commit()
