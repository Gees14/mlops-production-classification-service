"""Model training — supports logistic_regression, random_forest, and xgboost."""

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.logging import get_logger
from src.utils.paths import resolve, ensure_dir

logger = get_logger(__name__)


def build_model(model_type: str, random_seed: int = 42) -> BaseEstimator:
    """Instantiate a classifier by name.

    Supported types: logistic_regression, random_forest, xgboost.
    Falls back gracefully if xgboost is not installed.
    """
    model_type = model_type.lower()

    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_seed)

    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_seed)

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
            return XGBClassifier(
                n_estimators=100,
                random_state=random_seed,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        except ImportError:
            logger.warning("xgboost not installed; falling back to random_forest.")
            return RandomForestClassifier(n_estimators=100, random_state=random_seed)

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Choose from: logistic_regression, random_forest, xgboost."
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    random_seed: int = 42,
) -> BaseEstimator:
    """Train and return a fitted classifier."""
    model = build_model(model_type, random_seed)
    logger.info(f"Training {model.__class__.__name__} on {X_train.shape[0]} samples…")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def save_model(
    model: BaseEstimator,
    path: str = "artifacts/model/model.joblib",
) -> None:
    """Persist the trained model to disk."""
    full_path = resolve(path)
    ensure_dir(full_path.parent)
    joblib.dump(model, full_path)
    logger.info(f"Model saved to {full_path}")


def load_model(path: str = "artifacts/model/model.joblib") -> BaseEstimator:
    """Load a previously saved model."""
    full_path = resolve(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Model not found at {full_path}")
    model = joblib.load(full_path)
    logger.info(f"Model loaded from {full_path}")
    return model
