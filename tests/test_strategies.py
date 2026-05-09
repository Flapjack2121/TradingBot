import pandas as pd
import pytest

from data.indicators import compute_indicators
from strategies import REGISTRY, SignalEngine, registry_for_mode
from strategies.engine import RiskConfig
from strategies.base import BaseStrategy


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_strategy_returns_signal(name: str, ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(ohlcv)
    strat = REGISTRY[name]()
    sig = strat.generate("TEST", df)
    assert sig.ticker == "TEST"
    assert sig.strategy == name
    assert sig.side in {"BUY", "WAIT", "AVOID"}
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.rationale, "every signal must carry a rationale"
    assert isinstance(sig.rationale, str)


def test_engine_fills_risk(ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(ohlcv)
    engine = SignalEngine(
        strategies=[cls() for cls in REGISTRY.values()],
        risk=RiskConfig(account_size=10000, risk_per_trade=0.01,
                         atr_stop_multiplier=2.0, take_profit_r_multiple=2.0),
    )
    sigs = engine.evaluate("TEST", df)
    assert len(sigs) == len(REGISTRY)
    for s in sigs:
        if s.is_actionable:
            assert s.stop_loss is not None and s.stop_loss < s.entry
            assert s.take_profit is not None and s.take_profit > s.entry
            assert s.units > 0
            # check 2R take-profit
            assert abs((s.take_profit - s.entry) - 2 * (s.entry - s.stop_loss)) < 1e-6


def test_position_sizing_zero_when_invalid() -> None:
    pos = BaseStrategy.size_position(entry=100, stop_loss=110, account=10000, risk_pct=0.01)
    assert pos["units"] == 0
    pos = BaseStrategy.size_position(entry=100, stop_loss=95, account=10000, risk_pct=0.01)
    # Risk per unit = 5, risk amount = 100, so 20 units
    assert pos["units"] == pytest.approx(20.0)
    assert pos["risk_amount"] == 100.0


def test_engine_to_dataframe(ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(ohlcv)
    engine = SignalEngine(strategies=[cls() for cls in REGISTRY.values()])
    sigs = engine.scan({"X": df, "Y": df})
    out = engine.to_dataframe(sigs)
    assert {"Ticker", "Asset Class", "Strategy", "Signal", "Entry", "Stop Loss",
            "Take Profit", "Rationale"} <= set(out.columns)
    assert len(out) == 2 * len(REGISTRY)


def test_avoid_emitted_in_downtrend() -> None:
    """A monotonically falling series should produce AVOID for trend-filtered strategies."""
    import numpy as np
    n = 400
    close = np.linspace(200, 50, n)
    high = close * 1.005
    low = close * 0.995
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(n, 1_000_000.0)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    bear = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)
    df = compute_indicators(bear)
    # Trend-filtered swing strategies must AVOID in a confirmed downtrend.
    for name in ("minervini", "mean_reversion"):
        sig = REGISTRY[name]().generate("BEAR", df)
        assert sig.side == "AVOID", f"{name} should emit AVOID in pure downtrend"
        assert any(k in sig.rationale.lower() for k in ("down", "stage-2", "downtrend")), \
            f"{name} rationale must explain the AVOID"


def test_registry_for_mode_partitions_strategies() -> None:
    swing = registry_for_mode("swing")
    day = registry_for_mode("day")
    assert "minervini" in swing and "mean_reversion" in swing and "breakout" in swing
    assert "orb" in day and "vwap_reversion" in day and "inside_bar" in day
    # Mutually exclusive in this build
    assert set(swing).isdisjoint(set(day))


def test_every_strategy_has_modes_and_label() -> None:
    for name, cls in REGISTRY.items():
        assert cls.modes, f"{name} missing modes"
        assert cls.label, f"{name} missing label"
        assert isinstance(cls.description, str)


def test_every_strategy_declares_asset_classes_and_source() -> None:
    valid_classes = {
        "all", "Stocks US", "Stocks EU", "Stocks Asia",
        "Forex", "Commodity", "Crypto",
    }
    for name, cls in REGISTRY.items():
        assert cls.asset_classes, f"{name} missing asset_classes"
        for ac in cls.asset_classes:
            assert ac in valid_classes, f"{name} has invalid asset_class {ac!r}"
        assert cls.source, f"{name} missing source attribution"


def test_suits_universal_and_specific() -> None:
    from strategies import TurtleSystem, ConnorsRSI2
    assert TurtleSystem.suits("Forex")          # universal
    assert TurtleSystem.suits("Crypto")
    assert ConnorsRSI2.suits("Stocks US")       # equity-only
    assert not ConnorsRSI2.suits("Forex")
    assert not ConnorsRSI2.suits("Crypto")


def test_registry_for_asset_class_filters() -> None:
    from strategies import registry_for_asset_class
    forex_swing = registry_for_asset_class("Forex", mode_key="swing")
    # Universal swing systems should pass
    assert "breakout" in forex_swing       # Turtle
    assert "bb_breakout" in forex_swing    # Bollinger
    # Equity-only should be filtered out
    assert "minervini" not in forex_swing
    assert "mean_reversion" not in forex_swing


def test_apply_asset_classes(ohlcv: pd.DataFrame) -> None:
    df = compute_indicators(ohlcv)
    engine = SignalEngine(strategies=[cls() for cls in REGISTRY.values()])
    sigs = engine.scan({"AAPL": df, "EURUSD=X": df})
    SignalEngine.apply_asset_classes(sigs, {"AAPL": "Stocks US", "EURUSD=X": "Forex"})
    classes = {s.ticker: s.asset_class for s in sigs}
    assert classes["AAPL"] == "Stocks US"
    assert classes["EURUSD=X"] == "Forex"
