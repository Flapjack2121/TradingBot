"""Strategy interface and the :class:`Signal` value object.

Each concrete strategy returns a :class:`Signal` for the most recent bar of
the OHLCV+indicators DataFrame it is given. Signals carry everything the
dashboard needs to render the trade card: entry, stop loss, take profit,
position size in shares, risk amount, R-multiple, the contributing
indicators, and the strategy that produced it.

A strategy receives the *full* indicator DataFrame for one symbol; it is the
strategy's job to inspect the latest row(s) and decide whether to fire.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd


SIDE_BUY = "BUY"
SIDE_WAIT = "WAIT"
SIDE_AVOID = "AVOID"
VALID_SIDES = {SIDE_BUY, SIDE_WAIT, SIDE_AVOID}


@dataclass
class Signal:
    ticker: str
    strategy: str
    side: str              # "BUY", "WAIT", or "AVOID" (long-only for now)
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    atr: float | None = None
    confidence: float = 0.0    # 0..1, based on how many sub-conditions were met
    units: float = 0.0
    risk_amount: float = 0.0
    position_value: float = 0.0
    r_multiple: float | None = None
    reasons: dict[str, bool] = field(default_factory=dict)
    rationale: str = ""        # human-readable one-liner explaining the call
    asset_class: str = ""      # e.g. "Stocks US", "Forex", "Commodity"
    extras: dict[str, Any] = field(default_factory=dict)
    as_of: pd.Timestamp | None = None

    @property
    def is_actionable(self) -> bool:
        return (
            self.side == SIDE_BUY
            and self.entry is not None
            and self.stop_loss is not None
        )

    @property
    def is_avoid(self) -> bool:
        return self.side == SIDE_AVOID

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(d.get("as_of"), pd.Timestamp):
            d["as_of"] = d["as_of"].isoformat()
        return d


class BaseStrategy(ABC):
    """All strategies must implement :meth:`generate`."""

    name: str = "base"

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        """Return the signal for the latest bar of ``df``."""

    # ── helpers shared by all strategies ──────────────────────────────────
    @staticmethod
    def size_position(entry: float, stop_loss: float, account: float, risk_pct: float) -> dict:
        """Fixed-fractional position sizing.

        Returns a dict with units, risk_amount and position_value. If the
        stop is not below entry, returns zeros.
        """
        risk_per_unit = entry - stop_loss
        if risk_per_unit <= 0 or account <= 0 or risk_pct <= 0:
            return {"units": 0.0, "risk_amount": 0.0, "position_value": 0.0}
        risk_amount = account * risk_pct
        units = risk_amount / risk_per_unit
        return {
            "units": round(units, 4),
            "risk_amount": round(risk_amount, 2),
            "position_value": round(units * entry, 2),
        }

    @staticmethod
    def take_profit_from_r(entry: float, stop_loss: float, r: float) -> float:
        return round(entry + (entry - stop_loss) * r, 4)

    @staticmethod
    def _empty(ticker: str, name: str, reason: str = "insufficient data") -> Signal:
        return Signal(
            ticker=ticker, strategy=name, side=SIDE_WAIT,
            reasons={reason: False},
            rationale=f"No signal — {reason}.",
        )
