"""Dataset validation — produces a structured report before training."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import resolve, ensure_dir

logger = get_logger(__name__)


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
    min_rows: int = 20,
    max_missing_ratio: float = 0.3,
    max_class_imbalance_ratio: float = 10.0,
) -> dict[str, Any]:
    """Run validation checks on a loaded DataFrame.

    Checks:
    - Target column exists
    - Minimum row count
    - Missing value ratio per column
    - Duplicate rows
    - Class imbalance in target
    - Number of unique target classes (expects 2+ for classification)

    Args:
        df: Input DataFrame.
        target_column: Name of the label column.
        min_rows: Minimum acceptable number of rows.
        max_missing_ratio: Maximum fraction of missing values per column.
        max_class_imbalance_ratio: Max ratio of majority/minority class count.

    Returns:
        Validation report dict. ``report["passed"]`` is True only if no
        blocking checks fail.
    """
    issues: list[str] = []
    warnings: list[str] = []

    # --- Target column ---
    if target_column not in df.columns:
        issues.append(f"Target column '{target_column}' not found in dataset.")
        return _build_report(df, target_column, issues, warnings, passed=False)

    # --- Row count ---
    if len(df) < min_rows:
        issues.append(f"Dataset has only {len(df)} rows; minimum is {min_rows}.")

    # --- Missing values ---
    missing = df.isnull().mean()
    high_missing = missing[missing > max_missing_ratio]
    if not high_missing.empty:
        for col, ratio in high_missing.items():
            issues.append(f"Column '{col}' has {ratio:.1%} missing values (threshold {max_missing_ratio:.0%}).")

    # --- Duplicate rows ---
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"{dup_count} duplicate rows detected.")

    # --- Class distribution ---
    class_counts = df[target_column].value_counts()
    unique_classes = class_counts.index.tolist()

    if len(unique_classes) < 2:
        issues.append(f"Target column has only {len(unique_classes)} unique class(es); classification requires ≥ 2.")
    else:
        majority = class_counts.iloc[0]
        minority = class_counts.iloc[-1]
        imbalance_ratio = majority / minority if minority > 0 else float("inf")
        if imbalance_ratio > max_class_imbalance_ratio:
            warnings.append(
                f"Class imbalance ratio {imbalance_ratio:.1f}x exceeds threshold {max_class_imbalance_ratio:.0f}x. "
                "Consider resampling or class-weighted training."
            )

    passed = len(issues) == 0
    report = _build_report(df, target_column, issues, warnings, passed)
    _log_report(report)
    return report


def _build_report(
    df: pd.DataFrame,
    target_column: str,
    issues: list[str],
    warnings: list[str],
    passed: bool,
) -> dict[str, Any]:
    class_dist: dict[str, int] = {}
    if target_column in df.columns:
        class_dist = df[target_column].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

    missing_summary = {
        col: round(float(ratio), 4)
        for col, ratio in df.isnull().mean().items()
        if ratio > 0
    }

    return {
        "passed": passed,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "target_column": target_column,
        "class_distribution": class_dist,
        "missing_value_ratios": missing_summary,
        "duplicate_rows": int(df.duplicated().sum()),
        "issues": issues,
        "warnings": warnings,
    }


def _log_report(report: dict[str, Any]) -> None:
    status = "PASSED" if report["passed"] else "FAILED"
    logger.info(f"Data validation {status} — {len(report['issues'])} issue(s), {len(report['warnings'])} warning(s)")
    for issue in report["issues"]:
        logger.error(f"  [ISSUE] {issue}")
    for warning in report["warnings"]:
        logger.warning(f"  [WARN] {warning}")


def save_validation_report(report: dict[str, Any], output_path: str = "reports/data_validation.json") -> None:
    """Persist the validation report to JSON."""
    path = resolve(output_path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved to {path}")
