"""Prediction logging and monitoring summary."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import load_config, get_nested
from src.utils.logging import get_logger
from src.utils.paths import resolve, ensure_dir

logger = get_logger(__name__)


def _get_log_path() -> Path:
    try:
        cfg = load_config()
        rel = get_nested(cfg, "paths", "prediction_log_path", default="logs/predictions.jsonl")
    except Exception:
        rel = "logs/predictions.jsonl"
    path = resolve(rel)
    ensure_dir(path.parent)
    return path


def log_prediction(
    features: dict[str, Any],
    prediction: str,
    confidence: float,
    model_version: str,
    drift_warnings: list[str] | None = None,
) -> None:
    """Append a single prediction record to the JSONL log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_features": features,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version,
        "drift_warnings": drift_warnings or [],
    }
    path = _get_log_path()
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_monitoring_summary() -> dict[str, Any]:
    """Read the prediction log and return aggregated monitoring statistics."""
    path = _get_log_path()

    if not path.exists():
        return {
            "total_predictions": 0,
            "average_confidence": 0.0,
            "prediction_distribution": {},
            "latest_prediction_timestamp": None,
            "drift_warning_count": 0,
        }

    records: list[dict[str, Any]] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except json.JSONDecodeError as exc:
        logger.warning(f"Corrupted prediction log entry: {exc}")

    if not records:
        return {
            "total_predictions": 0,
            "average_confidence": 0.0,
            "prediction_distribution": {},
            "latest_prediction_timestamp": None,
            "drift_warning_count": 0,
        }

    distribution: dict[str, int] = defaultdict(int)
    confidences: list[float] = []
    drift_warning_count = 0

    for r in records:
        distribution[str(r.get("prediction", "unknown"))] += 1
        conf = r.get("confidence", 0.0)
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))
        drift_warning_count += len(r.get("drift_warnings", []))

    latest_ts = records[-1].get("timestamp") if records else None
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "total_predictions": len(records),
        "average_confidence": avg_confidence,
        "prediction_distribution": dict(distribution),
        "latest_prediction_timestamp": latest_ts,
        "drift_warning_count": drift_warning_count,
    }
