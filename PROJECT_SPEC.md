# Project Specification — MLOps Production Classification Service

## Product Goal

Provide a reusable, config-driven ML classification service that can be retargeted to any tabular binary-classification problem by changing a single YAML file and dataset. Designed to demonstrate production MLOps practices rather than notebook-style experimentation.

## Target Users

- ML Engineers building or evaluating production ML systems.
- AI Engineers adding classification capabilities to applications.
- MLOps practitioners assessing pipeline patterns (tracking, logging, drift detection).
- Portfolio reviewers evaluating ML engineering candidates.

## User Stories

| As a… | I want to… | So that… |
|---|---|---|
| ML Engineer | Replace the sample dataset with real data and retrain | I can classify production records without code changes |
| AI Engineer | Send features to `/predict` and receive a label + confidence | I can integrate the service into a downstream application |
| MLOps Practitioner | View experiment runs in MLflow UI | I can compare model versions across training runs |
| Team Lead | Run `make test` in CI | I know the pipeline is not broken before merging |
| Data Scientist | Read `reports/model_card.md` | I understand the model's performance and limitations |

## Core Features

1. Config-driven training — all paths, model type, and feature lists in `configs/config.yaml`.
2. Data validation with structured JSON report.
3. Scikit-learn ColumnTransformer preprocessing pipeline.
4. Supported models: logistic regression, random forest, optional XGBoost.
5. MLflow experiment tracking (parameters, metrics, artifacts).
6. FastAPI inference API with Pydantic validation.
7. Batch prediction endpoint.
8. JSONL prediction logging.
9. Monitoring summary endpoint.
10. Heuristic drift detection (mean deviation + unseen categories).
11. Docker and docker-compose deployment.
12. GitHub Actions CI workflow.

## Non-Functional Requirements

- Training must complete on the sample dataset in under 60 seconds on a modern laptop.
- API response time for a single prediction must be under 500ms on a local machine.
- No external services required for local development (MLflow runs locally).
- Project must work without XGBoost installed.
- All configuration via files and environment variables — no hardcoded values.

## API Requirements

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns service liveness and model-loaded status |
| `/model/info` | GET | Returns model type, version, and feature lists |
| `/predict` | POST | Single-record inference with drift warnings |
| `/predict/batch` | POST | Multi-record inference |
| `/monitor/log-prediction` | POST | Manual prediction logging |
| `/monitor/summary` | GET | Aggregated prediction statistics |

## Monitoring Requirements

- All predictions logged to `logs/predictions.jsonl`.
- Each log entry includes: timestamp, input features, prediction, confidence, model_version, drift_warnings.
- `/monitor/summary` aggregates: total predictions, average confidence, prediction distribution, latest timestamp, drift warning count.
- Drift detection warns (does not block) on numeric features deviating > 2σ from training mean, or on unseen categorical values.

## Success Criteria

- `make test` passes without a trained model artifact.
- `make train` completes successfully on the sample dataset.
- `make run-api` starts the API and `/health` returns 200.
- `/predict` returns a valid prediction with confidence.
- `/monitor/summary` returns aggregated stats after predictions.
- MLflow UI shows at least one completed run after training.
- Docker container builds and starts successfully.

## Out-of-Scope Features

- Multi-class classification beyond binary (can be added without structural changes).
- Online/incremental learning.
- Feature store integration.
- Automated retraining on drift detection.
- Cloud deployment (AWS/GCP/Azure).
- Authentication or API key management.
- Real-time streaming inference.
- Advanced statistical drift tests (KS test, PSI).

## Limitations

- The included dataset is **entirely synthetic** and for pipeline validation only.
- Drift detection is heuristic, not statistically rigorous.
- Prediction log grows unbounded — add rotation for production use.
- MLflow runs are stored locally; no authentication or remote artifact store.
- No multi-class probability calibration.
