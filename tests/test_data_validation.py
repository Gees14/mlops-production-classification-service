"""Tests for src/data/validation.py"""

import pandas as pd

from src.data.validation import validate_dataset


def test_valid_dataset_passes(sample_df):
    report = validate_dataset(sample_df, "churned", min_rows=5)
    assert report["passed"] is True
    assert len(report["issues"]) == 0


def test_missing_target_column_fails(sample_df):
    report = validate_dataset(sample_df, "nonexistent_col")
    assert report["passed"] is False
    assert any("not found" in issue for issue in report["issues"])


def test_too_few_rows_fails():
    df = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
    report = validate_dataset(df, "label", min_rows=10)
    assert report["passed"] is False
    assert any("rows" in issue for issue in report["issues"])


def test_high_missing_values_fails():
    df = pd.DataFrame(
        {
            "a": [None, None, None, None, 5.0],
            "b": [1, 2, 3, 4, 5],
            "label": [0, 1, 0, 1, 0],
        }
    )
    report = validate_dataset(df, "label", min_rows=3, max_missing_ratio=0.2)
    assert report["passed"] is False
    assert any("missing" in issue for issue in report["issues"])


def test_class_distribution_recorded(sample_df):
    report = validate_dataset(sample_df, "churned")
    dist = report["class_distribution"]
    assert "0" in dist or 0 in dist
    assert "1" in dist or 1 in dist


def test_duplicate_rows_warning():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "label": [0, 0, 1],
        }
    )
    report = validate_dataset(df, "label", min_rows=2)
    assert any("duplicate" in w.lower() for w in report["warnings"])
