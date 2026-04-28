"""Standalone evaluation script — loads saved artifacts and re-evaluates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.features.preprocessing import load_preprocessor
from src.models.train import load_model
from src.models.evaluate import (
    compute_metrics,
    save_metrics,
    save_confusion_matrix_figure,
    generate_model_card,
)
from src.utils.config import load_config, get_nested
from src.utils.logging import get_logger

logger = get_logger("evaluate")


def main() -> None:
    cfg = load_config()

    dataset_path = get_nested(cfg, "dataset", "dataset_path")
    target_column = get_nested(cfg, "dataset", "target_column")
    model_type = get_nested(cfg, "model", "model_type", default="random_forest")
    model_version = get_nested(cfg, "model", "model_version", default="local-dev")
    model_path = get_nested(cfg, "paths", "model_path", default="artifacts/model/model.joblib")
    preprocessor_path = get_nested(cfg, "paths", "preprocessor_path", default="artifacts/preprocessing/preprocessor.joblib")

    logger.info("=== Loading artifacts ===")
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)

    logger.info("=== Loading dataset ===")
    df = load_csv(dataset_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_t = preprocessor.transform(X)

    logger.info("=== Computing metrics ===")
    metrics = compute_metrics(model, X_t, y.values)
    save_metrics(metrics)
    save_confusion_matrix_figure(
        metrics["confusion_matrix"],
        class_names=[str(c) for c in sorted(y.unique())],
    )
    generate_model_card(
        metrics, model_type, dataset_path, target_column, model_version
    )

    logger.info("=== Evaluation complete ===")
    logger.info(f"  Accuracy : {metrics['accuracy']}")
    logger.info(f"  F1       : {metrics['f1']}")
    logger.info(f"  ROC-AUC  : {metrics.get('roc_auc', 'n/a')}")


if __name__ == "__main__":
    main()
