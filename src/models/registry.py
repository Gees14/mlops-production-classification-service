"""MLflow experiment and run management."""

from typing import Any

from src.utils.logging import get_logger
from src.utils.paths import resolve

logger = get_logger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure the MLflow tracking server and set the active experiment."""
    try:
        import mlflow

        _REMOTE_SCHEMES = ("http://", "https://", "databricks", "sqlite:///", "postgresql", "mysql", "mssql")
        if any(tracking_uri.startswith(s) for s in _REMOTE_SCHEMES):
            uri = tracking_uri
        else:
            # Plain relative paths (e.g. "mlruns") need a file:// URI on Windows
            # so MLflow doesn't mistake the drive letter for an unknown scheme.
            uri = resolve(tracking_uri).as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: '{experiment_name}' at '{tracking_uri}'")
    except ImportError:
        logger.warning("mlflow not installed — experiment tracking disabled.")


def log_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str],
    run_name: str = "training-run",
) -> None:
    """Log a complete training run to MLflow.

    Args:
        params: Hyperparameters and config values.
        metrics: Scalar metrics to log.
        artifacts: List of local file paths to upload as artifacts.
        run_name: Human-readable run label.
    """
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)

            scalar_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            mlflow.log_metrics(scalar_metrics)

            for artifact_path in artifacts:
                full = resolve(artifact_path)
                if full.exists():
                    mlflow.log_artifact(str(full))
                else:
                    logger.warning(f"Artifact not found, skipping: {full}")

        logger.info(f"MLflow run '{run_name}' logged successfully.")
    except ImportError:
        logger.warning("mlflow not installed — skipping run logging.")
    except Exception as exc:
        logger.error(f"MLflow logging failed: {exc}")
