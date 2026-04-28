"""Training entry-point: data → validate → preprocess → train → evaluate → MLflow."""

import sys
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_csv
from src.data.validation import validate_dataset, save_validation_report
from src.features.preprocessing import fit_preprocessor, save_preprocessor
from src.models.train import train_model, save_model
from src.models.evaluate import compute_metrics, save_metrics, save_confusion_matrix_figure, generate_model_card
from src.models.registry import setup_mlflow, log_run
from src.monitoring.drift import compute_training_stats, save_training_stats
from src.utils.config import load_config, get_nested
from src.utils.logging import get_logger

logger = get_logger("train")


def main() -> None:
    cfg = load_config()

    dataset_path = get_nested(cfg, "dataset", "dataset_path")
    target_column = get_nested(cfg, "dataset", "target_column")
    test_size = get_nested(cfg, "dataset", "test_size", default=0.2)
    random_seed = get_nested(cfg, "dataset", "random_seed", default=42)
    model_type = get_nested(cfg, "model", "model_type", default="random_forest")
    model_version = get_nested(cfg, "model", "model_version", default="local-dev")
    numeric_features = get_nested(cfg, "features", "numeric_features") or None
    categorical_features = get_nested(cfg, "features", "categorical_features") or None

    # Paths
    model_path = get_nested(cfg, "paths", "model_path", default="artifacts/model/model.joblib")
    preprocessor_path = get_nested(cfg, "paths", "preprocessor_path", default="artifacts/preprocessing/preprocessor.joblib")
    training_stats_path = get_nested(cfg, "paths", "training_stats_path", default="artifacts/preprocessing/training_stats.json")

    # MLflow
    mlflow_uri = get_nested(cfg, "mlflow", "mlflow_tracking_uri", default="mlruns")
    experiment_name = get_nested(cfg, "mlflow", "experiment_name", default="classification-service")

    # Validation settings
    min_rows = get_nested(cfg, "validation", "min_rows", default=20)
    max_missing_ratio = get_nested(cfg, "validation", "max_missing_ratio", default=0.3)
    max_class_imbalance_ratio = get_nested(cfg, "validation", "max_class_imbalance_ratio", default=10.0)

    # --- Load ---
    logger.info("=== Step 1: Load data ===")
    df = load_csv(dataset_path)

    # --- Validate ---
    logger.info("=== Step 2: Validate data ===")
    report = validate_dataset(
        df, target_column,
        min_rows=min_rows,
        max_missing_ratio=max_missing_ratio,
        max_class_imbalance_ratio=max_class_imbalance_ratio,
    )
    save_validation_report(report)
    if not report["passed"]:
        logger.error("Data validation failed. Aborting training.")
        sys.exit(1)

    # --- Split ---
    logger.info("=== Step 3: Train/test split ===")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Preprocess ---
    logger.info("=== Step 4: Fit preprocessing ===")
    preprocessor, num_cols, cat_cols = fit_preprocessor(
        X_train,
        target_column=None,  # X_train has no target column; pass None
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    save_preprocessor(preprocessor, preprocessor_path)

    # --- Training stats for drift detection ---
    stats = compute_training_stats(X_train, num_cols, cat_cols)
    save_training_stats(stats, training_stats_path)

    # --- Train ---
    logger.info("=== Step 5: Train model ===")
    model = train_model(X_train_t, y_train.values, model_type=model_type, random_seed=random_seed)
    save_model(model, model_path)

    # --- Evaluate ---
    logger.info("=== Step 6: Evaluate model ===")
    metrics = compute_metrics(model, X_test_t, y_test.values)
    save_metrics(metrics)
    save_confusion_matrix_figure(
        metrics["confusion_matrix"],
        class_names=[str(c) for c in sorted(y.unique())],
    )
    generate_model_card(
        metrics, model_type, dataset_path, target_column, model_version
    )

    # --- MLflow ---
    logger.info("=== Step 7: Log to MLflow ===")
    setup_mlflow(mlflow_uri, experiment_name)
    log_run(
        params={
            "model_type": model_type,
            "model_version": model_version,
            "test_size": test_size,
            "random_seed": random_seed,
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
            "numeric_features": ",".join(num_cols),
            "categorical_features": ",".join(cat_cols),
        },
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        artifacts=[model_path, preprocessor_path, "reports/data_validation.json", "reports/metrics.json"],
        run_name=f"{model_type}-{model_version}",
    )

    logger.info("=== Training complete ===")
    logger.info(f"  Accuracy : {metrics['accuracy']}")
    logger.info(f"  F1       : {metrics['f1']}")
    logger.info(f"  ROC-AUC  : {metrics.get('roc_auc', 'n/a')}")


if __name__ == "__main__":
    main()
