"""Tests for prediction request validation and monitoring."""

import json
import pytest
from pathlib import Path

from app.monitoring import log_prediction, get_monitoring_summary


def test_log_and_summarise(tmp_path, monkeypatch):
    """log_prediction writes a record; get_monitoring_summary reads it back."""
    log_file = tmp_path / "predictions.jsonl"

    # Redirect the monitoring module to use a temp log path
    import app.monitoring as mon_module

    original_get_log_path = mon_module._get_log_path
    monkeypatch.setattr(mon_module, "_get_log_path", lambda: log_file)

    log_prediction(
        features={"age": 35, "region": "east"},
        prediction="0",
        confidence=0.82,
        model_version="local-dev",
    )

    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["prediction"] == "0"
    assert record["confidence"] == 0.82

    summary = get_monitoring_summary()
    assert summary["total_predictions"] == 1
    assert summary["average_confidence"] == 0.82
    assert "0" in summary["prediction_distribution"]


def test_summary_empty_log(tmp_path, monkeypatch):
    import app.monitoring as mon_module

    nonexistent = tmp_path / "empty.jsonl"
    monkeypatch.setattr(mon_module, "_get_log_path", lambda: nonexistent)

    summary = get_monitoring_summary()
    assert summary["total_predictions"] == 0
    assert summary["average_confidence"] == 0.0


def test_drift_check_numeric_warning():
    from src.monitoring.drift import check_drift

    training_stats = {
        "numeric": {"age": {"mean": 35.0, "std": 5.0, "min": 20.0, "max": 60.0}},
        "categorical": {},
    }
    result = check_drift({"age": 100}, training_stats, mean_deviation_threshold=2.0)
    assert result["has_warnings"] is True
    assert any("age" in w for w in result["warnings"])


def test_drift_check_unseen_category():
    from src.monitoring.drift import check_drift

    training_stats = {
        "numeric": {},
        "categorical": {"region": {"known_categories": ["north", "south", "east"]}},
    }
    result = check_drift({"region": "mars"}, training_stats)
    assert result["has_warnings"] is True
    assert any("region" in w for w in result["warnings"])


def test_drift_check_no_warning():
    from src.monitoring.drift import check_drift

    training_stats = {
        "numeric": {"age": {"mean": 35.0, "std": 5.0, "min": 20.0, "max": 60.0}},
        "categorical": {"region": {"known_categories": ["north", "south"]}},
    }
    result = check_drift({"age": 36, "region": "north"}, training_stats)
    assert result["has_warnings"] is False
