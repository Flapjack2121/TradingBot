"""Multi-ticker, multi-year strategy validation.

The :class:`StrategyLab` runs a strategy across an entire basket of tickers
over a long history (5–20+ years), then aggregates the per-ticker results
into one portfolio view: combined equity curve, pooled trade statistics,
and risk-adjusted performance metrics.

Each ticker is backtested independently with the *same* starting cash. The
"portfolio equity curve" is the sum of per-ticker equity curves over the
union of dates (forward-filled where a symbol has no data yet). This models
an equally-weighted basket where capital is committed per symbol up to the
budget configured per ticker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from data import DataManager
from data.indicators import compute_indicators
from strategies.base import BaseStrategy

from .engine import Backtester, BacktestResult

log = logging.getLogger(__name__)


@dataclass
class LabResult:
    strategy_name: str
    period: str
    initial_cash_per_ticker: float
    interval: str = "1d"
    per_ticker: dict[str, BacktestResult] = field(default_factory=dict)
    portfolio_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    portfolio_stats: dict = field(default_factory=dict)
    pooled_trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def per_ticker_table(self) -> pd.DataFrame:
        rows = []
        for ticker, res in self.per_ticker.items():
            row = {"ticker": ticker, **res.stats}
            row["final_equity"] = res.final_equity
            rows.append(row)
        return pd.DataFrame(rows).sort_values("total_return_pct", ascending=False) \
                                  if rows else pd.DataFrame()


@dataclass
class StrategyLab:
    data_manager: DataManager
    initial_cash_per_ticker: float = 10_000.0
    risk_per_trade: float = 0.01
    atr_stop_multiplier: float = 2.0
    take_profit_r_multiple: float = 2.0
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0005
    max_hold_bars: int = 60
    long_cache_ttl_hours: float = 24 * 7   # weekly refresh is plenty for backtests

    # ── public ────────────────────────────────────────────────────────────
    def run(self, strategy: BaseStrategy, tickers: Iterable[str],
            period: str = "10y", interval: str = "1d",
            indicator_cfg: dict | None = None, progress=None) -> LabResult:
        bt = Backtester(
            initial_cash=self.initial_cash_per_ticker,
            risk_per_trade=self.risk_per_trade,
            atr_stop_multiplier=self.atr_stop_multiplier,
            take_profit_r_multiple=self.take_profit_r_multiple,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            max_hold_bars=self.max_hold_bars,
        )
        ind_cfg = indicator_cfg or self.data_manager.indicator_cfg

        per_ticker: dict[str, BacktestResult] = {}
        tickers = list(tickers)
        total = len(tickers)
        for i, t in enumerate(tickers):
            if progress:
                progress(i, total, t)
            try:
                df = self.data_manager.get(
                    t, force_refresh=False, period=period, interval=interval,
                    ttl_hours=self.long_cache_ttl_hours,
                )
            except Exception as exc:
                log.warning("Skipping %s (download failed): %s", t, exc)
                continue
            if df is None or df.empty or len(df) < 250:
                continue
            try:
                df = compute_indicators(df, ind_cfg)
                per_ticker[t] = bt.run(t, df, strategy)
            except Exception as exc:
                log.warning("Backtest failed for %s: %s", t, exc)
                continue

        if progress:
            progress(total, total, "")

        portfolio_eq = self._aggregate_equity(per_ticker)
        pooled_trades = self._pool_trades(per_ticker)
        portfolio_stats = Backtester._summarize(portfolio_eq, pooled_trades)
        portfolio_stats["tickers_run"] = len(per_ticker)
        portfolio_stats["initial_cash_total"] = self.initial_cash_per_ticker * len(per_ticker)
        portfolio_stats["final_equity_total"] = (
            float(portfolio_eq.iloc[-1]) if not portfolio_eq.empty else 0.0
        )

        return LabResult(
            strategy_name=strategy.name,
            period=period,
            interval=interval,
            initial_cash_per_ticker=self.initial_cash_per_ticker,
            per_ticker=per_ticker,
            portfolio_equity=portfolio_eq,
            portfolio_stats=portfolio_stats,
            pooled_trades=pooled_trades,
        )

    # ── helpers ───────────────────────────────────────────────────────────
    def _aggregate_equity(self, results: dict[str, BacktestResult]) -> pd.Series:
        if not results:
            return pd.Series(dtype=float)
        # Build a wide DataFrame: dates × tickers; equity per ticker.
        # ffill so a ticker that hasn't started trading yet still contributes
        # its initial cash to the portfolio total.
        eq_frames = {}
        for t, r in results.items():
            if r.equity_curve is None or r.equity_curve.empty:
                continue
            eq_frames[t] = r.equity_curve
        if not eq_frames:
            return pd.Series(dtype=float)
        wide = pd.DataFrame(eq_frames)
        wide = wide.sort_index().ffill()
        # Pre-IPO bars get filled with the initial cash so the basket size is constant.
        wide = wide.fillna(self.initial_cash_per_ticker)
        # Drop duplicate timestamps (e.g. holidays returned multiple times)
        wide = wide[~wide.index.duplicated(keep="last")]
        return wide.sum(axis=1)

    @staticmethod
    def _pool_trades(results: dict[str, BacktestResult]) -> pd.DataFrame:
        frames = [r.trades for r in results.values() if not r.trades.empty]
        if not frames:
            return pd.DataFrame()
        pooled = pd.concat(frames, ignore_index=True)
        return pooled.sort_values("entry_time").reset_index(drop=True)
