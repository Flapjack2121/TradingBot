"""Pytest fixtures and shared test utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def synthetic_ohlcv(n: int = 400, seed: int = 7, trend: float = 0.0005) -> pd.DataFrame:
    """Generate a synthetic OHLCV series with mild upward drift."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close * (1 - rng.uniform(0.001, 0.015, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv()
