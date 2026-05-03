from data.indicators import compute_indicators
from backtest import Backtester
from strategies import TripleConfirmation, MeanReversion, DonchianBreakout


def test_backtest_runs(ohlcv) -> None:
    df = compute_indicators(ohlcv)
    bt = Backtester(initial_cash=10_000, max_hold_bars=20)
    result = bt.run("X", df, MeanReversion())
    assert result.initial_cash == 10_000
    assert not result.equity_curve.empty
    # equity curve should contain at most the available bars (one entry per iteration plus final)
    assert len(result.equity_curve) <= len(df) + 1
    assert "trades" in result.stats


def test_backtest_no_signals_breakeven(ohlcv) -> None:
    """A strategy that never fires should leave equity equal to initial cash."""
    df = compute_indicators(ohlcv)

    class Never(TripleConfirmation):
        def generate(self, ticker, df):  # type: ignore[override]
            sig = super().generate(ticker, df)
            sig.side = "WAIT"
            sig.entry = None
            return sig

    bt = Backtester(initial_cash=10_000)
    result = bt.run("X", df, Never())
    assert result.stats["trades"] == 0
    assert result.final_equity == 10_000
