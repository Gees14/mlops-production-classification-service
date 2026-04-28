"""Project path utilities — all paths resolved relative to project root."""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the project root directory (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def resolve(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return project_root() / relative_path


def ensure_dir(path: str | Path) -> Path:
    """Create directory and all parents if they do not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
