# Tasks — MLOps Production Classification Service

## Data Tasks
- [x] Create synthetic sample dataset (`data/sample_data.csv`, 60 rows)
- [x] Implement CSV loader with existence and format checks (`src/data/loader.py`)
- [x] Implement dataset validation with structured JSON report (`src/data/validation.py`)
- [x] Write tests for loader and validation (`tests/test_data_loader.py`, `tests/test_data_validation.py`)

## Training Tasks
- [x] Implement auto-detecting ColumnTransformer preprocessing (`src/features/preprocessing.py`)
- [x] Implement model builder supporting logistic_regression, random_forest, xgboost (`src/models/train.py`)
- [x] Implement evaluation metrics and reporting (`src/models/evaluate.py`)
- [x] Implement confusion matrix figure generation (`src/models/evaluate.py`)
- [x] Implement model card generation (`src/models/evaluate.py`)
- [x] Implement training stats computation for drift (`src/monitoring/drift.py`)
- [x] Implement MLflow experiment logging (`src/models/registry.py`)
- [x] Write full training pipeline script (`scripts/train.py`)
- [x] Write standalone evaluation script (`scripts/evaluate.py`)
- [x] Write training smoke test (`tests/test_training_smoke.py`)

## API Tasks
- [x] Define Pydantic request/response schemas (`app/schemas.py`)
- [x] Implement inference singleton with graceful load failure (`app/predictor.py`)
- [x] Implement FastAPI application with all required endpoints (`app/main.py`)
- [x] Implement GET /health
- [x] Implement GET /model/info
- [x] Implement POST /predict
- [x] Implement POST /predict/batch
- [x] Implement POST /monitor/log-prediction
- [x] Implement GET /monitor/summary
- [x] Write API health tests (`tests/test_api_health.py`)
- [x] Write API server startup script (`scripts/run_api.py`)

## Monitoring Tasks
- [x] Implement JSONL prediction logging (`app/monitoring.py`)
- [x] Implement monitoring summary aggregation (`app/monitoring.py`)
- [x] Implement heuristic drift detection (`src/monitoring/drift.py`)
- [x] Write monitoring and drift tests (`tests/test_prediction_validation.py`)

## DevOps Tasks
- [x] Write Dockerfile
- [x] Write docker-compose.yml
- [x] Write Makefile with all required targets
- [x] Write GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- [x] Write .gitignore
- [x] Write .env.example

## Testing Tasks
- [x] `tests/test_data_loader.py`
- [x] `tests/test_data_validation.py`
- [x] `tests/test_preprocessing.py`
- [x] `tests/test_training_smoke.py`
- [x] `tests/test_api_health.py`
- [x] `tests/test_prediction_validation.py`

## Documentation Tasks
- [x] CLAUDE.md — project guide for AI assistants
- [x] PROJECT_SPEC.md — product specification
- [x] ARCHITECTURE.md — system architecture with Mermaid diagrams
- [x] TASKS.md — this file
- [x] DECISIONS.md — architecture decision records
- [x] README.md — professional portfolio README
- [x] data/README.md — dataset documentation

## Future Improvements

- [ ] Add multiclass support (precision/recall per class, macro averaging)
- [ ] Add KS-test or Population Stability Index (PSI) for rigorous drift detection
- [ ] Add async inference with background prediction logging
- [ ] Add model versioning with MLflow Model Registry
- [ ] Add Kubernetes deployment manifests
- [ ] Add Prometheus metrics endpoint for observability
- [ ] Add feature importance endpoint
- [ ] Add automated retraining trigger on drift threshold
- [ ] Add cloud deployment (AWS SageMaker / GCP Vertex AI)
- [ ] Add authentication (API key or JWT)
- [ ] Add prediction log rotation to prevent unbounded growth
