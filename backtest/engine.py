"""Lightweight bar-by-bar backtester.

This is a deliberately simple, dependency-free engine — no vectorbt required.
It walks the price series chronologically, asks the strategy for a signal at
each bar, and simulates an entry on the next open with an ATR stop and an R-
multiple take-profit. Commissions and slippage are applied symmetrically.

The goal is to validate signal logic and produce equity curves and summary
stats — not to be a tick-accurate institutional simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    stats: dict
    initial_cash: float

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if not self.equity_curve.empty else self.initial_cash

    @property
    def total_return_pct(self) -> float:
        return float((self.final_equity / self.initial_cash - 1) * 100)


@dataclass
class Backtester:
    initial_cash: float = 10000.0
    risk_per_trade: float = 0.01
    atr_stop_multiplier: float = 2.0
    take_profit_r_multiple: float = 2.0
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0005
    max_hold_bars: int = 60

    def run(self, ticker: str, df: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        if df is None or df.empty:
            return BacktestResult(pd.DataFrame(), pd.Series(dtype=float), {}, self.initial_cash)

        cash = self.initial_cash
        equity: list[float] = []
        timestamps: list[pd.Timestamp] = []
        trades: list[dict] = []

        position: dict | None = None
        # We need future bar to fill the next-open entry, so iterate up to len-1.
        for i in range(len(df) - 1):
            bar = df.iloc[i]
            window = df.iloc[: i + 1]

            # mark-to-market equity
            current_equity = cash
            if position is not None:
                current_equity += position["units"] * float(bar["Close"])
            equity.append(current_equity)
            timestamps.append(df.index[i])

            # exit logic for open position
            if position is not None:
                next_bar = df.iloc[i + 1]
                exit_price, exit_reason = self._maybe_exit(position, next_bar, i + 1)
                if exit_price is not None:
                    cash += self._close(position, exit_price, df.index[i + 1], exit_reason, trades)
                    position = None
                    continue

            # entry logic
            if position is None:
                sig = strategy.generate(ticker, window)
                if sig.is_actionable:
                    next_bar = df.iloc[i + 1]
                    fill = float(next_bar["Open"]) * (1 + self.slippage_pct)
                    atr_val = sig.atr if sig.atr else float(bar.get("atr", 0.0))
                    if atr_val and atr_val > 0:
                        stop = fill - atr_val * self.atr_stop_multiplier
                        tp = fill + (fill - stop) * self.take_profit_r_multiple
                        risk_amount = current_equity * self.risk_per_trade
                        units = risk_amount / (fill - stop)
                        cost = units * fill * (1 + self.commission_pct)
                        if cost <= cash and units > 0:
                            cash -= cost
                            position = {
                                "ticker": ticker,
                                "strategy": strategy.name,
                                "entry": fill,
                                "stop": stop,
                                "tp": tp,
                                "units": units,
                                "entry_time": df.index[i + 1],
                                "entry_index": i + 1,
                                "atr": atr_val,
                            }

        # close any remaining position at the last close
        last_close = float(df.iloc[-1]["Close"])
        equity.append(cash + (position["units"] * last_close if position else 0))
        timestamps.append(df.index[-1])
        if position is not None:
            cash += self._close(position, last_close, df.index[-1], "eod", trades)
            position = None

        eq = pd.Series(equity, index=pd.Index(timestamps, name="date"))
        eq = eq[~eq.index.duplicated(keep="last")]
        trades_df = pd.DataFrame(trades)
        stats = self._summarize(eq, trades_df)
        return BacktestResult(trades_df, eq, stats, self.initial_cash)

    # ── helpers ───────────────────────────────────────────────────────────
    def _maybe_exit(self, pos: dict, next_bar: pd.Series, idx: int) -> tuple[float | None, str]:
        high = float(next_bar["High"])
        low = float(next_bar["Low"])
        # if the bar gapped past stop or tp at open, use open price
        nxt_open = float(next_bar["Open"])
        if nxt_open <= pos["stop"]:
            return nxt_open * (1 - self.slippage_pct), "stop_gap"
        if nxt_open >= pos["tp"]:
            return nxt_open * (1 - self.slippage_pct), "tp_gap"
        # intraday touches — assume worst-case: stop checked first
        if low <= pos["stop"]:
            return pos["stop"] * (1 - self.slippage_pct), "stop"
        if high >= pos["tp"]:
            return pos["tp"] * (1 - self.slippage_pct), "tp"
        # time-based exit
        if (idx - pos["entry_index"]) >= self.max_hold_bars:
            return float(next_bar["Close"]) * (1 - self.slippage_pct), "time"
        return None, ""

    def _close(self, pos: dict, exit_price: float, ts: pd.Timestamp, reason: str, trades: list[dict]) -> float:
        proceeds = pos["units"] * exit_price * (1 - self.commission_pct)
        pnl = pos["units"] * (exit_price - pos["entry"])
        risk_per_unit = pos["entry"] - pos["stop"]
        r = (exit_price - pos["entry"]) / risk_per_unit if risk_per_unit > 0 else 0.0
        trades.append({
            "ticker": pos["ticker"],
            "strategy": pos["strategy"],
            "entry_time": pos["entry_time"],
            "exit_time": ts,
            "entry": pos["entry"],
            "exit": exit_price,
            "stop": pos["stop"],
            "tp": pos["tp"],
            "units": pos["units"],
            "pnl": round(pnl, 2),
            "r": round(r, 3),
            "reason": reason,
        })
        return proceeds

    @staticmethod
    def _summarize(eq: pd.Series, trades: pd.DataFrame) -> dict:
        if eq.empty:
            return {}
        ret = eq.pct_change().dropna()
        peak = eq.cummax()
        dd = (eq / peak - 1)
        sharpe = float(np.sqrt(252) * ret.mean() / ret.std()) if ret.std() else 0.0
        max_dd = float(dd.min()) if not dd.empty else 0.0
        wins = trades[trades["pnl"] > 0] if not trades.empty else trades
        losses = trades[trades["pnl"] <= 0] if not trades.empty else trades
        return {
            "trades": int(len(trades)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if len(trades) else 0.0,
            "total_return_pct": round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 3),
            "avg_R": round(float(trades["r"].mean()), 3) if not trades.empty else None,
            "expectancy": round(float(trades["pnl"].mean()), 2) if not trades.empty else None,
        }

    def run_many(self, frames: dict[str, pd.DataFrame], strategy: BaseStrategy) -> dict[str, BacktestResult]:
        return {t: self.run(t, df, strategy) for t, df in frames.items()}
