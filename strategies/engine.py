"""SignalEngine — orchestrates strategies across the universe."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .base import BaseStrategy, Signal

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    atr_stop_multiplier: float = 2.0
    take_profit_r_multiple: float = 2.0


@dataclass
class SignalEngine:
    strategies: list[BaseStrategy]
    risk: RiskConfig = field(default_factory=RiskConfig)

    def evaluate(self, ticker: str, df: pd.DataFrame) -> list[Signal]:
        results: list[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.generate(ticker, df)
            except Exception as exc:
                log.warning("Strategy %s failed on %s: %s", strat.name, ticker, exc)
                continue
            self._fill_risk(sig)
            results.append(sig)
        return results

    def scan(self, frames: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        for ticker, df in frames.items():
            signals.extend(self.evaluate(ticker, df))
        return signals

    def actionable(self, signals: Iterable[Signal]) -> list[Signal]:
        return [s for s in signals if s.is_actionable]

    # ── helpers ───────────────────────────────────────────────────────────
    def _fill_risk(self, sig: Signal) -> None:
        if sig.entry is None or sig.atr is None or sig.atr <= 0:
            return
        stop_dist = sig.atr * self.risk.atr_stop_multiplier
        sig.stop_loss = round(sig.entry - stop_dist, 4)
        sig.take_profit = BaseStrategy.take_profit_from_r(
            sig.entry, sig.stop_loss, self.risk.take_profit_r_multiple
        )
        pos = BaseStrategy.size_position(
            sig.entry, sig.stop_loss, self.risk.account_size, self.risk.risk_per_trade
        )
        sig.units = pos["units"]
        sig.risk_amount = pos["risk_amount"]
        sig.position_value = pos["position_value"]
        sig.r_multiple = self.risk.take_profit_r_multiple

    @staticmethod
    def to_dataframe(signals: Iterable[Signal]) -> pd.DataFrame:
        rows = []
        for s in signals:
            rows.append({
                "Ticker": s.ticker,
                "Strategy": s.strategy,
                "Signal": s.side,
                "Confidence": round(s.confidence, 2),
                "Entry": s.entry,
                "Stop Loss": s.stop_loss,
                "Take Profit": s.take_profit,
                "ATR": round(s.atr, 4) if s.atr else None,
                "Units": s.units,
                "Risk €": s.risk_amount,
                "Position €": s.position_value,
                "R": s.r_multiple,
                "As Of": s.as_of,
            })
        return pd.DataFrame(rows)
