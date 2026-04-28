"""Pydantic request and response schemas for the inference API."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Feature key-value pairs matching the columns used during training.",
        examples=[{"age": 42, "monthly_spend": 120.5, "region": "north", "account_type": "premium"}],
    )


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    features_received: dict[str, Any]
    drift_warnings: list[str] = Field(default_factory=list)


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        description="List of feature dicts, one per record.",
        min_length=1,
    )


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total: int


class LogPredictionRequest(BaseModel):
    features: dict[str, Any]
    prediction: str
    confidence: float
    model_version: str


class MonitorSummaryResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    prediction_distribution: dict[str, int]
    latest_prediction_timestamp: Optional[str]
    drift_warning_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
