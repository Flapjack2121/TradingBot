"""Smoke tests for the two newly added strategies — Bollinger Squeeze
Breakout (swing) and Pivot Point Breakout (day)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.indicators import compute_indicators
from strategies import BollingerBandBreakout, PivotPointBreakout, SIDE_BUY, SIDE_WAIT, SIDE_AVOID


def _daily_ohlcv(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.001, 0.012, n))
    low = close * (1 - rng.uniform(0.001, 0.012, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _hourly_ohlcv(n: int = 600, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.005, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.0005, 0.005, n))
    low = close * (1 - rng.uniform(0.0005, 0.005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="h")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_bb_breakout_returns_signal() -> None:
    df = compute_indicators(_daily_ohlcv())
    sig = BollingerBandBreakout().generate("TEST", df)
    assert sig.strategy == "bb_breakout"
    assert sig.side in {SIDE_BUY, SIDE_WAIT, SIDE_AVOID}
    assert sig.rationale


def test_pivot_breakout_returns_signal() -> None:
    df = compute_indicators(_hourly_ohlcv())
    sig = PivotPointBreakout().generate("TEST", df)
    assert sig.strategy == "pivot_breakout"
    assert sig.side in {SIDE_BUY, SIDE_WAIT, SIDE_AVOID}
    assert sig.rationale


def test_pivot_extras_contain_classic_levels() -> None:
    df = compute_indicators(_hourly_ohlcv())
    sig = PivotPointBreakout().generate("TEST", df)
    if sig.side != SIDE_WAIT or "pivot" in sig.extras:
        for key in ("pivot", "R1", "R2", "S1", "S2"):
            assert key in sig.extras, f"missing pivot level {key}"
