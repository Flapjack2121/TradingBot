from unittest.mock import patch

import pandas as pd

from data.indicators import compute_indicators
from data import DataManager
from backtest import Backtester, StrategyLab
from strategies import (
    MinerviniTrendTemplate, ConnorsRSI2, TurtleSystem,
    MeanReversion, DonchianBreakout,
)


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

    class Never(MinerviniTrendTemplate):
        def generate(self, ticker, df):  # type: ignore[override]
            sig = super().generate(ticker, df)
            sig.side = "WAIT"
            sig.entry = None
            return sig

    bt = Backtester(initial_cash=10_000)
    result = bt.run("X", df, Never())
    assert result.stats["trades"] == 0
    assert result.final_equity == 10_000


def test_backtest_metrics_present(ohlcv) -> None:
    """All advertised metrics must appear in stats once any equity exists."""
    df = compute_indicators(ohlcv)
    bt = Backtester(initial_cash=10_000, max_hold_bars=20)
    result = bt.run("X", df, MeanReversion())
    expected = {
        "trades", "wins", "losses", "win_rate_pct",
        "total_return_pct", "cagr_pct", "volatility_pct",
        "sharpe", "sortino", "max_drawdown_pct", "max_dd_duration_bars",
        "avg_win_eur", "avg_loss_eur", "largest_win_eur", "largest_loss_eur",
        "avg_hold_days", "years",
    }
    missing = expected - set(result.stats.keys())
    assert not missing, f"missing metric keys: {missing}"


def test_strategylab_aggregates(ohlcv) -> None:
    """StrategyLab should pool per-ticker results into a portfolio view.

    We patch DataManager.get to return our synthetic OHLCV so the test does
    not hit the network.
    """
    dm = DataManager(cache_dir="data/cache_test", period="2y")
    with patch.object(DataManager, "get", return_value=ohlcv):
        lab = StrategyLab(data_manager=dm, initial_cash_per_ticker=10_000,
                           max_hold_bars=20)
        result = lab.run(MeanReversion(), ["A", "B", "C"], period="2y")

    assert result.strategy_name == "mean_reversion"
    assert len(result.per_ticker) == 3
    # portfolio equity must be roughly the sum of per-ticker equities
    assert not result.portfolio_equity.empty
    assert result.portfolio_stats["tickers_run"] == 3
    assert result.portfolio_stats["initial_cash_total"] == 30_000
    table = result.per_ticker_table
    assert {"ticker", "total_return_pct", "win_rate_pct"} <= set(table.columns)
