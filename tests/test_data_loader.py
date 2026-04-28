"""Tests for src/data/loader.py"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import load_csv


def test_load_sample_csv():
    df = load_csv("data/sample_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "churned" in df.columns


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("data/nonexistent_file.csv")


def test_load_returns_correct_shape():
    df = load_csv("data/sample_data.csv")
    assert df.shape[1] > 1, "Expected multiple columns"


def test_load_tmp_csv(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,label\n1,x,0\n2,y,1\n")
    df = load_csv(str(csv_file))
    assert list(df.columns) == ["a", "b", "label"]
    assert len(df) == 2
