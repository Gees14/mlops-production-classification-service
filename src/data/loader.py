"""CSV data loading with basic existence and format checks."""

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import resolve

logger = get_logger(__name__)


def load_csv(dataset_path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame.

    Args:
        dataset_path: Path relative to project root or absolute path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    path = Path(dataset_path)
    if not path.is_absolute():
        path = resolve(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV at {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df
