"""Smoke tests for the day-trading strategies on synthetic hourly bars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.indicators import compute_indicators
from strategies import (
    OpeningRangeBreakout, VWAPReversion, InsideBarBreakout,
    SIDE_BUY, SIDE_WAIT, SIDE_AVOID,
)


def _hourly_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly bars spanning many session days, US trading hours."""
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


def test_orb_returns_signal() -> None:
    df = compute_indicators(_hourly_ohlcv())
    sig = OpeningRangeBreakout().generate("TEST", df)
    assert sig.strategy == "orb"
    assert sig.side in {SIDE_BUY, SIDE_WAIT, SIDE_AVOID}
    assert sig.rationale


def test_vwap_reversion_returns_signal() -> None:
    df = compute_indicators(_hourly_ohlcv())
    sig = VWAPReversion().generate("TEST", df)
    assert sig.strategy == "vwap_reversion"
    assert sig.side in {SIDE_BUY, SIDE_WAIT, SIDE_AVOID}
    assert sig.rationale


def test_inside_bar_returns_signal() -> None:
    df = compute_indicators(_hourly_ohlcv())
    sig = InsideBarBreakout().generate("TEST", df)
    assert sig.strategy == "inside_bar"
    assert sig.side in {SIDE_BUY, SIDE_WAIT, SIDE_AVOID}
    assert sig.rationale


def test_inside_bar_buy_on_constructed_pattern() -> None:
    """Build a tiny series with an inside bar then a breakout — should fire BUY."""
    # bars: 0 (wide), 1 (inside), 2 (breakout). Plus padding for ATR(14).
    n = 30
    base = np.full(n, 100.0)
    high = base * 1.005
    low = base * 0.995

    # Bar -3 (ref): wide range 95-105
    high[-3] = 105.0
    low[-3] = 95.0
    base[-3] = 100.0
    # Bar -2 (inside): nested inside ref → 96-101
    high[-2] = 101.0
    low[-2] = 96.0
    base[-2] = 99.0
    # Bar -1 (breakout): close above inside high
    high[-1] = 103.0
    low[-1] = 100.5
    base[-1] = 102.5

    df = pd.DataFrame({
        "Open": np.roll(base, 1),
        "High": high, "Low": low, "Close": base,
        "Volume": np.full(n, 500_000.0),
    }, index=pd.date_range("2024-06-01", periods=n, freq="h"))
    df = compute_indicators(df)

    # Disable NR4 requirement so we test the IB+breakout core
    sig = InsideBarBreakout({"require_nr4": False}).generate("TEST", df)
    assert sig.side == SIDE_BUY, f"expected BUY, got {sig.side}: {sig.rationale}"
