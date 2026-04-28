"""Simple feature drift detection based on training statistics.

Limitations:
- Uses only mean deviation for numeric features (not a full distribution test).
- Categorical drift checks for unseen categories only.
- This is a heuristic early-warning system, not a rigorous statistical test.
  Consider scipy KS-test or evidently for production-grade drift detection.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import resolve, ensure_dir

logger = get_logger(__name__)

_DEFAULT_STATS_PATH = "artifacts/preprocessing/training_stats.json"


def compute_training_stats(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    """Compute and return training feature statistics for drift comparison."""
    stats: dict[str, Any] = {"numeric": {}, "categorical": {}}

    for col in numeric_features:
        if col in df.columns:
            stats["numeric"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

    for col in categorical_features:
        if col in df.columns:
            stats["categorical"][col] = {
                "known_categories": df[col].dropna().unique().tolist(),
            }

    return stats


def save_training_stats(
    stats: dict[str, Any],
    path: str = _DEFAULT_STATS_PATH,
) -> None:
    """Persist training statistics to JSON."""
    full_path = resolve(path)
    ensure_dir(full_path.parent)
    with open(full_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Training stats saved to {full_path}")


def load_training_stats(path: str = _DEFAULT_STATS_PATH) -> dict[str, Any] | None:
    """Load training statistics if available."""
    full_path = resolve(path)
    if not full_path.exists():
        return None
    with open(full_path, "r") as f:
        return json.load(f)


def check_drift(
    features: dict[str, Any],
    training_stats: dict[str, Any],
    mean_deviation_threshold: float = 2.0,
) -> dict[str, Any]:
    """Compare a single prediction's features against training statistics.

    Args:
        features: Dict of feature name → value from the inference request.
        training_stats: Stats produced by compute_training_stats().
        mean_deviation_threshold: Number of std-devs above which numeric
            deviation triggers a warning.

    Returns:
        Dict with ``warnings`` list and ``has_warnings`` bool.
    """
    warnings: list[str] = []

    for col, val in features.items():
        if col in training_stats.get("numeric", {}):
            col_stats = training_stats["numeric"][col]
            std = col_stats["std"]
            if std and std > 0:
                deviation = abs(float(val) - col_stats["mean"]) / std
                if deviation > mean_deviation_threshold:
                    warnings.append(
                        f"Feature '{col}' value {val} deviates {deviation:.1f}σ from training mean "
                        f"({col_stats['mean']:.2f} ± {std:.2f})."
                    )

        if col in training_stats.get("categorical", {}):
            known = training_stats["categorical"][col]["known_categories"]
            if str(val) not in [str(k) for k in known]:
                warnings.append(
                    f"Feature '{col}' has unseen category '{val}' (not in training set)."
                )

    return {"warnings": warnings, "has_warnings": len(warnings) > 0}
