"""Config loading and access utilities."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.paths import project_root

load_dotenv()

_DEFAULT_CONFIG_PATH = project_root() / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Falls back to configs/config.yaml."""
    path = Path(config_path or os.getenv("CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Allow environment variable overrides for MLflow tracking URI
    if os.getenv("MLFLOW_TRACKING_URI"):
        cfg.setdefault("mlflow", {})["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    return cfg


def get_nested(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely access nested config keys."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node
