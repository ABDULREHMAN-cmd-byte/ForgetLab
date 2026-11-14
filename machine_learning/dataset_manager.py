"""Utilities for validating, importing, and preprocessing datasets.

The module ensures uploaded CSV files are safe and compatible with the ML
workflow before they are used to train models or evaluate unlearning.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]


def _normalize_frame(frame: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """Trim source whitespace and convert UCI question-mark sentinels to NaN."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        cleaned = frame[column].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
        frame[column] = cleaned.astype(object).where(cleaned.notna(), float("nan"))
    if target_column and target_column in frame.columns:
        frame[target_column] = frame[target_column].astype(str)
    return frame


def load_uci_adult_dataset(
    training_path: str | os.PathLike[str],
    testing_path: str | os.PathLike[str],
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Load both official UCI files and preserve their original train/test boundary."""
    training_frame = pd.read_csv(
        training_path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
    )
    testing_frame = pd.read_csv(
        testing_path,
        header=None,
        names=ADULT_COLUMNS,
        comment="|",
        skipinitialspace=True,
    )
    if training_frame.shape[1] != len(ADULT_COLUMNS) or testing_frame.shape[1] != len(ADULT_COLUMNS):
        raise ValueError("The UCI Adult files must each contain 15 columns.")

    # UCI appends a period to labels in adult.test; remove only that suffix.
    testing_frame["income"] = testing_frame["income"].astype("string").str.replace(r"\.$", "", regex=True)
    training_frame = _normalize_frame(training_frame, "income")
    testing_frame = _normalize_frame(testing_frame, "income")
    training_count = len(training_frame)
    combined = pd.concat([training_frame, testing_frame], ignore_index=True)
    return combined, list(range(training_count)), list(range(training_count, len(combined)))


def sanitize_filename(filename: str) -> str:
    """Return a safe filename while removing path traversal and special characters.

    Args:
        filename: Raw uploaded filename.

    Returns:
        A cleaned filename that is safe for storage on disk.
    """
    cleaned = os.path.basename(filename)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    return cleaned or "dataset.csv"


def validate_csv_file(path: str | os.PathLike[str]) -> bool:
    """Validate that a CSV exists, is readable, and contains a tabular structure.

    Args:
        path: File path to inspect.

    Returns:
        True when validation succeeds; otherwise False.
    """
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return False
    if file_path.suffix.lower() != ".csv":
        return False
    try:
        frame = pd.read_csv(file_path)
    except Exception:
        return False
    return isinstance(frame, pd.DataFrame) and not frame.empty


def ensure_adult_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign UCI Adult column names to raw datasets that omit headers.

    Args:
        frame: Data imported from the Adult dataset source.

    Returns:
        A DataFrame with canonical Adult column names.
    """
    if list(frame.columns) == [f"Unnamed: {i}" for i in range(len(frame.columns))] or len(frame.columns) == len(ADULT_COLUMNS):
        renamed = frame.copy()
        renamed.columns = ADULT_COLUMNS[: len(renamed.columns)]
        return renamed
    return frame


def detect_target_column(frame: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    """Choose a likely target from a dataset column list.

    Args:
        frame: The DataFrame to inspect.
        preferred: Optional preferred target name.

    Returns:
        The selected target column or None when no suitable target is found.
    """
    candidate_names = ["income", "label", "target", "class", "y"]
    for name in [preferred] + candidate_names:
        if name and name in frame.columns:
            return name
    return frame.columns[-1] if len(frame.columns) > 0 else None


def load_dataset(path: str | os.PathLike[str], target_column: Optional[str] = None, base_dir: Optional[str] = None) -> pd.DataFrame:
    """Load and normalize a dataset file for model training.

    The function handles CSV files, rows with '?', and raw Adult census data
    with no headers. It also ensures the target column is present.

    Args:
        path: Dataset path to read.
        target_column: Optional explicit target column name.
        base_dir: Optional base directory for resolving relative paths.

    Returns:
        A cleaned DataFrame ready for preprocessing.
    """
    file_path = Path(path)
    frame = pd.read_csv(file_path)
    frame = ensure_adult_headers(frame)
    frame = _normalize_frame(frame, target_column)
    if target_column is None:
        target_column = detect_target_column(frame)
    if target_column and target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    return frame


class DatasetManager:
    """High-level dataset helper that validates and stores uploads."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.upload_dir = self.base_dir / "dataset" / "uploads"
        self.processed_dir = self.base_dir / "dataset" / "processed"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_csv(self, uploaded_file) -> str:
        """Persist an uploaded CSV to a safe location and return the final path."""
        safe_name = sanitize_filename(uploaded_file.filename)
        destination = self.upload_dir / safe_name
        uploaded_file.save(str(destination))
        return str(destination)

    def validate_dataset(self, path: str) -> bool:
        """Check whether a dataset file is valid and ready for training."""
        return validate_csv_file(path)

    def load_dataset(self, path: str, target_column: Optional[str] = None) -> pd.DataFrame:
        """Load a dataset from disk and normalize it for ML use."""
        return load_dataset(path, target_column=target_column)
