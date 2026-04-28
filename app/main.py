"""FastAPI application — thin routing layer only."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.predictor import predictor
from app.monitoring import log_prediction, get_monitoring_summary
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    LogPredictionRequest,
    ModelInfoResponse,
    MonitorSummaryResponse,
    PredictRequest,
    PredictResponse,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="MLOps Production Classification Service",
    description=(
        "Config-driven tabular classification with MLflow tracking, "
        "prediction logging, and basic drift detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(status="ok", model_loaded=predictor.is_loaded)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded model."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")
    return ModelInfoResponse(
        model_type=predictor.model_type,
        model_version=predictor.model_version,
        target_column=predictor.target_column,
        numeric_features=predictor.numeric_features,
        categorical_features=predictor.categorical_features,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Run inference on a single feature dict and log the prediction."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")
    try:
        result = predictor.predict(request.features)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=422, detail="Prediction failed. Check that features match training columns.")

    log_prediction(
        features=request.features,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        drift_warnings=result["drift_warnings"],
    )

    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        features_received=request.features,
        drift_warnings=result["drift_warnings"],
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Run inference on a list of feature dicts."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")

    responses: list[PredictResponse] = []
    for features in request.records:
        try:
            result = predictor.predict(features)
        except Exception as exc:
            logger.error(f"Batch prediction error: {exc}")
            raise HTTPException(status_code=422, detail=f"Prediction failed for record: {features}")

        log_prediction(
            features=features,
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            drift_warnings=result["drift_warnings"],
        )
        responses.append(
            PredictResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                model_version=result["model_version"],
                features_received=features,
                drift_warnings=result["drift_warnings"],
            )
        )

    return BatchPredictResponse(predictions=responses, total=len(responses))


@app.post("/monitor/log-prediction", tags=["Monitoring"])
def manual_log_prediction(request: LogPredictionRequest) -> dict[str, str]:
    """Manually log a prediction (useful for external systems)."""
    log_prediction(
        features=request.features,
        prediction=request.prediction,
        confidence=request.confidence,
        model_version=request.model_version,
    )
    return {"status": "logged"}


@app.get("/monitor/summary", response_model=MonitorSummaryResponse, tags=["Monitoring"])
def monitor_summary() -> MonitorSummaryResponse:
    """Return aggregated prediction log statistics."""
    summary = get_monitoring_summary()
    return MonitorSummaryResponse(**summary)
