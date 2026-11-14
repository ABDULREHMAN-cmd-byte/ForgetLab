"""Application configuration and runtime settings for ForgetLab.

This module centralizes settings such as the SQLite database path, upload
locations, and general Flask defaults. Keeping configuration in a single place
makes the app easier to test and run in different environments.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("FORGETLAB_DATA_DIR", str(BASE_DIR)))


class Config:
    """Strongly typed Flask config values for the web application."""

    SECRET_KEY = os.getenv("FORGETLAB_SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "FORGETLAB_DATABASE_URI",
        f"sqlite:///{DATA_DIR / 'forgetlab.db'}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = str(DATA_DIR / "dataset" / "uploads")
    PROCESSED_DATASET_FOLDER = str(DATA_DIR / "dataset" / "processed")
    MODEL_ORIGINAL_FOLDER = str(DATA_DIR / "models" / "original")
    MODEL_UNLEARNED_FOLDER = str(DATA_DIR / "models" / "unlearned")
    RESULTS_FOLDER = str(DATA_DIR / "results")
    GRAPHS_FOLDER = str(DATA_DIR / "results" / "graphs")
    REPORTS_FOLDER = str(DATA_DIR / "results" / "reports")
    ALLOWED_UPLOAD_EXTENSIONS = {"csv", "xlsx", "xls"}
    DEFAULT_TARGET_COLUMN = "income"
    DEFAULT_MODEL = "logistic_regression"


CONFIG = Config()
