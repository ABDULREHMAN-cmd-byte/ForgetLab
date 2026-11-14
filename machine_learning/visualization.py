"""Visualization utilities for plotting model comparison metrics.

The module produces charts for the dashboard and experiment reports, using
Matplotlib and JSON-safe metric payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metric_comparison(labels: List[str], original_values: List[float], unlearned_values: List[float], title: str, output_path: str) -> str:
    """Create a bar chart comparing original and unlearned model metrics.

    Args:
        labels: Metric names.
        original_values: Original model scores.
        unlearned_values: Unlearned model scores.
        title: Plot title.
        output_path: Destination image path.

    Returns:
        The path to the generated chart.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], original_values, width=0.4, label="Original")
    ax.bar([i + 0.2 for i in x], unlearned_values, width=0.4, label="Unlearned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
