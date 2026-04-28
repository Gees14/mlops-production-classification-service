"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on the path regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame matching the sample dataset schema."""
    return pd.DataFrame(
        {
            "age": [28, 45, 33, 22, 52],
            "tenure_months": [6, 36, 12, 3, 60],
            "monthly_spend": [45.5, 120.0, 75.25, 30.0, 200.0],
            "num_products": [1, 3, 2, 1, 4],
            "num_support_tickets": [3, 0, 1, 5, 0],
            "region": ["north", "south", "east", "north", "west"],
            "account_type": ["basic", "premium", "standard", "basic", "premium"],
            "payment_method": ["credit_card", "bank_transfer", "credit_card", "credit_card", "bank_transfer"],
            "churned": [1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_features() -> dict:
    """Single inference feature dict."""
    return {
        "age": 35,
        "tenure_months": 18,
        "monthly_spend": 85.0,
        "num_products": 2,
        "num_support_tickets": 1,
        "region": "east",
        "account_type": "standard",
        "payment_method": "credit_card",
    }
