# Architecture Decision Records — MLOps Production Classification Service

## ADR-001: FastAPI for the Inference Service

**Decision:** Use FastAPI as the HTTP framework.

**Reasons:**
- Native Pydantic integration for request/response validation with minimal boilerplate.
- Automatic OpenAPI/Swagger UI generation at `/docs`.
- Async-capable without requiring async code everywhere.
- De-facto standard for Python ML inference APIs in 2024.

**Alternatives considered:**
- Flask: simpler but no built-in schema validation or async support.
- Django REST Framework: heavier, designed for full applications rather than microservices.

**Future migration:** FastAPI scales to production deployments behind nginx or a Kubernetes ingress without changes.

---

## ADR-002: MLflow for Experiment Tracking

**Decision:** Use MLflow for parameter, metric, and artifact logging.

**Reasons:**
- Local-first (writes to `mlruns/` with no external service required).
- UI available with a single CLI command.
- Compatible with remote tracking servers (Databricks, self-hosted) by changing one URI.
- Industry-standard for portfolio-level ML projects.

**Alternatives considered:**
- Weights & Biases: requires account and external service.
- DVC: focused on data versioning rather than experiment tracking.
- Custom JSON logging: sufficient for artifacts but provides no UI or comparison tooling.

**Future migration:** Set `MLFLOW_TRACKING_URI` to a remote MLflow server or Databricks workspace without code changes.

---

## ADR-003: JSONL Prediction Logging

**Decision:** Log predictions to a local JSONL file (`logs/predictions.jsonl`).

**Reasons:**
- No external dependency — works in any environment.
- Line-delimited JSON is trivially readable with standard tools.
- Easily replaceable: point log writes at a database, Kafka topic, or cloud storage by changing `app/monitoring.py`.

**Alternatives considered:**
- SQLite: more queryable but adds a dependency and migration concern.
- PostgreSQL / Redis: requires external services not appropriate for a local portfolio project.
- Cloud logging (BigQuery, CloudWatch): over-engineered for local demonstration.

**Limitation:** Log grows unbounded. Add rotation (e.g., `RotatingFileHandler` or an external sink) for production.

---

## ADR-004: Config-Driven Training

**Decision:** All dataset paths, feature lists, model type, and thresholds live in `configs/config.yaml`.

**Reasons:**
- Enables dataset replacement without touching source code.
- Supports environment-variable overrides for CI and Docker.
- Makes the project clearly reusable for fraud detection, churn, risk scoring, etc.

**Alternatives considered:**
- CLI argument parsing only: forces users to rebuild commands rather than edit a config.
- Hardcoded values: anti-pattern; breaks reusability.

---

## ADR-005: Scikit-learn ColumnTransformer Pipeline

**Decision:** Use `ColumnTransformer` with `StandardScaler` (numeric) and `OneHotEncoder` (categorical).

**Reasons:**
- Handles mixed-type tabular data in one serialisable object.
- `joblib.dump` persists both fitted scaler and encoder together.
- `handle_unknown="ignore"` on OneHotEncoder prevents crashes on unseen categories at inference.
- Auto-detection via `select_dtypes` means users can leave feature lists empty in config.

**Alternatives considered:**
- Manual preprocessing: more brittle, harder to serialise consistently.
- Feature-engine or category_encoders: additional dependencies not needed for this scope.

---

## ADR-006: Docker for Deployment

**Decision:** Provide a `Dockerfile` and `docker-compose.yml` for containerised deployment.

**Reasons:**
- Reproducible environment regardless of host OS.
- Volumes for `artifacts/`, `logs/`, and `mlruns/` persist data across container restarts.
- Demonstrates production deployment awareness.

**Alternatives considered:**
- Bare uvicorn: simpler but not portable.
- Kubernetes: future improvement; not required for a local portfolio demonstration.

---

## ADR-007: Heuristic Drift Detection

**Decision:** Implement a lightweight, statistics-based drift check rather than a full distribution test.

**Reasons:**
- No additional dependencies (no scipy, no evidently).
- Produces human-readable warnings rather than p-values, which are easier to explain.
- Returns warnings that are logged alongside predictions; does not block inference.

**Limitations acknowledged:**
- Mean deviation is not a rigorous test (no significance threshold, no distribution shape check).
- A single out-of-range prediction does not constitute distributional drift.
- For production: replace with scipy KS-test, Population Stability Index (PSI), or the `evidently` library.

**Alternatives considered:**
- evidently: excellent library but heavy dependency; out of scope for this portfolio project.
- scipy KS-test: rigorous but requires batching predictions, adding state management complexity.

---

## ADR-008: Optional XGBoost

**Decision:** XGBoost is not in `requirements.txt` but is supported if the user installs it separately.

**Reasons:**
- XGBoost has C++ build dependencies that cause installation failures on some CI environments.
- Making it optional keeps CI reliable and requirements minimal.
- Graceful fallback to `random_forest` means the project never fails due to an optional dependency.

**How to enable:** `pip install xgboost` then set `model.model_type: xgboost` in `configs/config.yaml`.
