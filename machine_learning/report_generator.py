"""Generate experiment summaries and downloadable reports for ForgetLab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def generate_experiment_summary(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Convert experiment results into a concise summary payload.

    Args:
        experiment: Dictionary of results from a training or unlearning run.

    Returns:
        A summary dictionary for UI display and CSV export.
    """
    summary = {
        "method": experiment.get("method", "unknown"),
        "original_accuracy": experiment.get("original_metrics", {}).get("accuracy"),
        "unlearned_accuracy": experiment.get("unlearned_metrics", {}).get("accuracy"),
    }
    return summary


def write_json_report(path: str, payload: Dict[str, Any]) -> str:
    """Write a JSON payload to a report path.

    Args:
        path: Destination file path.
        payload: Data to save.

    Returns:
        The saved file path.
    """
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(report_path)
