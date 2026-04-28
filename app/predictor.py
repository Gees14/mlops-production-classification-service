"""Model and preprocessor loading + inference logic."""

from typing import Any

import numpy as np
import pandas as pd

from src.features.preprocessing import load_preprocessor
from src.models.train import load_model
from src.monitoring.drift import load_training_stats, check_drift
from src.utils.config import load_config, get_nested
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    """Holds the loaded model and preprocessor for inference."""

    def __init__(self) -> None:
        self._model = None
        self._preprocessor = None
        self._training_stats: dict[str, Any] | None = None
        self._cfg: dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        """Load model, preprocessor, training stats, and config from disk."""
        try:
            self._cfg = load_config()
            model_path = get_nested(self._cfg, "paths", "model_path", default="artifacts/model/model.joblib")
            preprocessor_path = get_nested(self._cfg, "paths", "preprocessor_path", default="artifacts/preprocessing/preprocessor.joblib")
            training_stats_path = get_nested(self._cfg, "paths", "training_stats_path", default="artifacts/preprocessing/training_stats.json")

            self._model = load_model(model_path)
            self._preprocessor = load_preprocessor(preprocessor_path)
            self._training_stats = load_training_stats(training_stats_path)
            self._loaded = True
            logger.info("Predictor loaded successfully.")
        except FileNotFoundError as exc:
            logger.warning(f"Artifacts not found: {exc}. Run `make train` first.")
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return get_nested(self._cfg, "model", "model_version", default="unknown")

    @property
    def model_type(self) -> str:
        return get_nested(self._cfg, "model", "model_type", default="unknown")

    @property
    def target_column(self) -> str:
        return get_nested(self._cfg, "dataset", "target_column", default="unknown")

    @property
    def numeric_features(self) -> list[str]:
        if self._preprocessor is None:
            return []
        try:
            return self._preprocessor.transformers_[0][2]
        except (IndexError, AttributeError):
            return []

    @property
    def categorical_features(self) -> list[str]:
        if self._preprocessor is None:
            return []
        try:
            return self._preprocessor.transformers_[1][2]
        except (IndexError, AttributeError):
            return []

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference on a single feature dict.

        Returns prediction label, confidence, model_version, and drift warnings.
        """
        if not self._loaded:
            raise RuntimeError("Model artifacts not loaded. Run `make train` first.")

        df = pd.DataFrame([features])
        X_t = self._preprocessor.transform(df)

        label = str(self._model.predict(X_t)[0])
        confidence = 0.0
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X_t)[0]
            confidence = round(float(np.max(proba)), 4)

        drift_result = {"warnings": []}
        if self._training_stats:
            drift_result = check_drift(features, self._training_stats)

        return {
            "prediction": label,
            "confidence": confidence,
            "model_version": self.model_version,
            "drift_warnings": drift_result["warnings"],
        }


# Module-level singleton — shared across all API requests
predictor = Predictor()
