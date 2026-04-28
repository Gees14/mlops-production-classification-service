"""Tests for src/features/preprocessing.py"""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    detect_feature_columns,
    build_preprocessor,
    fit_preprocessor,
)


def test_detect_feature_columns_excludes_target(sample_df):
    num_cols, cat_cols = detect_feature_columns(sample_df, "churned")
    assert "churned" not in num_cols
    assert "churned" not in cat_cols


def test_detect_numeric_columns(sample_df):
    num_cols, _ = detect_feature_columns(sample_df, "churned")
    assert "age" in num_cols
    assert "monthly_spend" in num_cols


def test_detect_categorical_columns(sample_df):
    _, cat_cols = detect_feature_columns(sample_df, "churned")
    assert "region" in cat_cols
    assert "account_type" in cat_cols


def test_fit_preprocessor_returns_fitted(sample_df):
    X = sample_df.drop(columns=["churned"])
    preprocessor, num_cols, cat_cols = fit_preprocessor(X, target_column=None)
    assert preprocessor is not None
    assert len(num_cols) > 0


def test_transform_produces_array(sample_df):
    X = sample_df.drop(columns=["churned"])
    preprocessor, _, _ = fit_preprocessor(X, target_column=None)
    result = preprocessor.transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(X)


def test_explicit_feature_lists(sample_df):
    X = sample_df.drop(columns=["churned"])
    preprocessor, num_cols, cat_cols = fit_preprocessor(
        X,
        target_column=None,
        numeric_features=["age", "monthly_spend"],
        categorical_features=["region"],
    )
    assert num_cols == ["age", "monthly_spend"]
    assert cat_cols == ["region"]
