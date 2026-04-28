"""Model evaluation — metrics, confusion matrix, classification report."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger
from src.utils.paths import resolve, ensure_dir

logger = get_logger(__name__)


def compute_metrics(
    model: BaseEstimator,
    X: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics for the given split.

    Returns a dict with accuracy, precision, recall, f1, roc_auc (binary only),
    confusion_matrix, and classification_report.
    """
    y_pred = model.predict(X)

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    classes = np.unique(y_true)
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
        except Exception as exc:
            logger.warning(f"Could not compute ROC-AUC: {exc}")

    logger.info(
        f"Metrics — accuracy={metrics['accuracy']}, f1={metrics['f1']}, "
        f"roc_auc={metrics.get('roc_auc', 'n/a')}"
    )
    return metrics


def save_metrics(metrics: dict[str, Any], path: str = "reports/metrics.json") -> None:
    """Save metrics dict to JSON."""
    full_path = resolve(path)
    ensure_dir(full_path.parent)
    with open(full_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {full_path}")


def save_confusion_matrix_figure(
    cm: list[list[int]],
    class_names: list[str] | None = None,
    path: str = "reports/figures/confusion_matrix.png",
) -> None:
    """Save a confusion matrix heatmap using matplotlib/seaborn."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        full_path = resolve(path)
        ensure_dir(full_path.parent)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names or "auto",
            yticklabels=class_names or "auto",
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(full_path)
        plt.close(fig)
        logger.info(f"Confusion matrix figure saved to {full_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping confusion matrix figure.")


def generate_model_card(
    metrics: dict[str, Any],
    model_type: str,
    dataset_path: str,
    target_column: str,
    model_version: str,
    path: str = "reports/model_card.md",
) -> None:
    """Write a minimal model card in Markdown."""
    full_path = resolve(path)
    ensure_dir(full_path.parent)

    lines = [
        "# Model Card\n",
        f"**Model type:** {model_type}  ",
        f"**Model version:** {model_version}  ",
        f"**Target column:** {target_column}  ",
        f"**Dataset:** {dataset_path}  \n",
        "## Evaluation Metrics\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Accuracy | {metrics.get('accuracy', 'n/a')} |",
        f"| Precision | {metrics.get('precision', 'n/a')} |",
        f"| Recall | {metrics.get('recall', 'n/a')} |",
        f"| F1 Score | {metrics.get('f1', 'n/a')} |",
        f"| ROC-AUC | {metrics.get('roc_auc', 'n/a')} |\n",
        "## Limitations\n",
        "- Metrics are reported on the held-out test split only.",
        "- The included sample dataset is **synthetic**. Do not use these numbers to claim real-world performance.",
        "- Model has not been audited for fairness or bias.\n",
    ]

    with open(full_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Model card saved to {full_path}")
