"""Smoke test: run the full training pipeline end-to-end on the sample dataset."""

import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_csv
from src.data.validation import validate_dataset
from src.features.preprocessing import fit_preprocessor
from src.models.train import train_model, build_model
from src.models.evaluate import compute_metrics


def test_build_logistic_regression():
    model = build_model("logistic_regression")
    assert model is not None


def test_build_random_forest():
    model = build_model("random_forest")
    assert model is not None


def test_build_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("nonexistent_model")


def test_full_training_smoke():
    """End-to-end smoke: load → validate → preprocess → train → evaluate."""
    df = load_csv("data/sample_data.csv")

    report = validate_dataset(df, "churned")
    assert report["passed"] is True

    X = df.drop(columns=["churned"])
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor, _, _ = fit_preprocessor(X_train, target_column=None)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = train_model(X_train_t, y_train.values, model_type="random_forest")
    metrics = compute_metrics(model, X_test_t, y_test.values)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
