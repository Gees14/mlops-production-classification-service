"""Feature preprocessing — ColumnTransformer pipeline with auto-detection."""

from typing import Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logging import get_logger
from src.utils.paths import resolve, ensure_dir

logger = get_logger(__name__)

# Columns that should never be used as features (IDs, free text, etc.)
_DROP_PATTERNS = {"id", "_id", "customer_id", "user_id", "uuid", "index"}


def detect_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    """Auto-detect numeric and categorical feature columns if not provided.

    Columns matching ID-like patterns are excluded automatically.
    """
    exclude = {target_column}
    for col in df.columns:
        if any(pattern in col.lower() for pattern in _DROP_PATTERNS):
            exclude.add(col)

    candidate_cols = [c for c in df.columns if c not in exclude]

    if not numeric_features:
        numeric_features = df[candidate_cols].select_dtypes(include="number").columns.tolist()

    if not categorical_features:
        categorical_features = df[candidate_cols].select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer with standard scaling and one-hot encoding."""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([("scaler", StandardScaler())])
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline(
            [
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            ]
        )
        transformers.append(("categorical", cat_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No feature columns available for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_preprocessor(
    df: pd.DataFrame,
    target_column: Optional[str],
    numeric_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Fit a preprocessing pipeline on a feature DataFrame.

    Args:
        df: Feature DataFrame (may or may not include the target column).
        target_column: Target column name to exclude, or None if df is
            already features-only.
        numeric_features: Explicit numeric column list; auto-detected if empty.
        categorical_features: Explicit categorical column list; auto-detected if empty.

    Returns:
        Fitted ColumnTransformer, resolved numeric feature list,
        resolved categorical feature list.
    """
    # Use a sentinel that cannot match any real column when target is absent
    effective_target = target_column if target_column is not None else "__no_target__"
    num_cols, cat_cols = detect_feature_columns(
        df, effective_target, numeric_features, categorical_features
    )
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df
    preprocessor.fit(X)
    logger.info("Preprocessing pipeline fitted.")
    return preprocessor, num_cols, cat_cols


def save_preprocessor(
    preprocessor: ColumnTransformer,
    path: str = "artifacts/preprocessing/preprocessor.joblib",
) -> None:
    """Persist the fitted preprocessor to disk."""
    full_path = resolve(path)
    ensure_dir(full_path.parent)
    joblib.dump(preprocessor, full_path)
    logger.info(f"Preprocessor saved to {full_path}")


def load_preprocessor(path: str = "artifacts/preprocessing/preprocessor.joblib") -> ColumnTransformer:
    """Load a previously saved preprocessor."""
    full_path = resolve(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {full_path}")
    preprocessor = joblib.load(full_path)
    logger.info(f"Preprocessor loaded from {full_path}")
    return preprocessor
