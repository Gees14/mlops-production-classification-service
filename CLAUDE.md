# CLAUDE.md — MLOps Production Classification Service

## Project Purpose

A production-style ML classification service demonstrating the full MLOps lifecycle:
raw CSV → data validation → preprocessing → training → evaluation → MLflow tracking → model artifacts → FastAPI inference → prediction logging → monitoring → Docker deployment → CI tests.

Designed as a portfolio project for ML Engineer, AI Engineer, and MLOps roles. The service is intentionally generic: swap the dataset and config to support fraud detection, churn, medical classification, purchase-order routing, risk scoring, and more.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn, optional xgboost |
| Experiment tracking | MLflow |
| API | FastAPI + Pydantic + uvicorn |
| Config | PyYAML + python-dotenv |
| Serialisation | joblib |
| Testing | pytest + httpx |
| Containerisation | Docker + docker-compose |
| CI | GitHub Actions |

## Architecture Overview

```
CSV data
  └── loader.py          → load, check existence
  └── validation.py      → checks, report
  └── preprocessing.py   → ColumnTransformer (scale + OHE)
  └── train.py (script)  → split, fit, evaluate, MLflow log
  └── artifacts/         → model.joblib, preprocessor.joblib, training_stats.json
  └── app/main.py        → FastAPI endpoints
  └── app/predictor.py   → inference singleton
  └── app/monitoring.py  → JSONL logging + summary
  └── src/monitoring/drift.py → heuristic drift warnings
```

## Important Commands

```bash
make install       # Install Python dependencies
make train         # Run full training pipeline
make evaluate      # Re-evaluate saved artifacts
make run-api       # Start FastAPI dev server (localhost:8000)
make test          # Run pytest suite
make mlflow-ui     # Open MLflow UI (localhost:5000)
make docker-build  # Build Docker image
make docker-up     # Start Docker Compose
make docker-down   # Stop Docker Compose
make clean         # Remove generated artifacts and logs
```

## Coding Conventions

- Type hints on all public function signatures.
- Short docstrings explaining *why*, not *what*, on public functions.
- No hardcoded file paths — use `src/utils/paths.py` helpers.
- No secrets or credentials in source code — use `.env`.
- Keep `app/main.py` thin: routing only. Logic belongs in `app/predictor.py`, `app/monitoring.py`, and `src/`.
- Config values come from `configs/config.yaml` or environment variables.

## Testing Commands

```bash
pytest tests/ -v                    # All tests
pytest tests/test_training_smoke.py # Training pipeline smoke test
pytest tests/test_api_health.py     # API health tests
```

## Rules for Modifying This Project

1. Do not make this notebook-first. All logic lives in `.py` modules.
2. Do not fake metrics. Reported metrics must be computed from real model outputs.
3. Do not hardcode local paths. Use `resolve()` from `src/utils/paths.py`.
4. Do not hardcode secrets. Use `.env` / environment variables.
5. Keep training, evaluation, API, and monitoring in separate modules.
6. Do not remove tests unless replacing them with better tests.
7. Update `ARCHITECTURE.md` whenever the pipeline structure changes.
8. Update `TASKS.md` when tasks are completed or new ones added.

## Security Rules

- Never log raw prediction inputs containing PII in production.
- Never expose internal stack traces via the API (use generic error messages).
- Never commit `.env`, model artifacts, or large data files.
- Validate all external inputs via Pydantic schemas before processing.

## What Claude Must Never Do

- Introduce hardcoded absolute paths.
- Add notebook cells or `.ipynb` files to this repository.
- Claim the synthetic sample dataset reflects real-world performance.
- Remove validation checks without replacing them.
- Commit secrets, credentials, or API keys.

## Definition of Done

A feature or fix is complete when:
- Unit tests pass for new/modified code.
- `make test` exits with code 0.
- Relevant documentation is updated.
- No hardcoded paths or secrets introduced.
- Docker build succeeds if changes affect `Dockerfile` or dependencies.
