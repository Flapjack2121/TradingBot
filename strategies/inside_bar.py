"""Inside Bar / NR4 compression breakout — day-trading classic.

Source: Toby Crabel, *Day Trading with Short Term Price Patterns and Opening
Range Breakout* (1990). The "Narrow Range" family (NR4 / NR7) and the
"Inside Bar" pattern are the cornerstones of Crabel's volatility-expansion
playbook used by countless prop and futures desks.

Logic:

  - **Inside Bar (IB)**: the prior bar's high < the bar before's high AND
    the prior bar's low > the bar before's low. Volatility has compressed.
  - **NR4** (optional): the prior bar's range was the narrowest of the
    last 4 bars. Compression has been building.
  - **Breakout**: the current bar closes above the inside bar's high
    (long signal) or below it (avoid).

Verdicts:
- **BUY**   inside bar formed AND current close > inside-bar high (with
            NR4 confirmation if enabled).
- **AVOID** inside bar formed AND current close < inside-bar low.
- **WAIT**  no compression / range still loose.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class InsideBarBreakout(BaseStrategy):
    name = "inside_bar"
    label = "🪤 Inside-Bar / NR4 Breakout"
    description = (
        "Crabel-style compression breakout — long when current bar breaks "
        "above an inside bar (optionally also NR4)."
    )
    modes = ["day"]

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 6:
            return self._empty(ticker, self.name)

        require_nr4 = self.params.get("require_nr4", True)

        last = df.iloc[-1]
        ib = df.iloc[-2]   # the candidate inside bar
        ref = df.iloc[-3]  # the bar that should *contain* the IB

        try:
            price = float(last["Close"])
            ib_high = float(ib["High"])
            ib_low = float(ib["Low"])
            ref_high = float(ref["High"])
            ref_low = float(ref["Low"])
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)
        if pd.isna(atr_val):
            return self._empty(ticker, self.name, "ATR NaN")

        is_inside_bar = (ib_high < ref_high) and (ib_low > ref_low)

        # NR4: the IB's range is the narrowest of the last 4 bars.
        last_4 = df.iloc[-5:-1]
        ib_range = ib_high - ib_low
        narrowest_of_4 = float((last_4["High"] - last_4["Low"]).min())
        is_nr4 = abs(ib_range - narrowest_of_4) < 1e-9

        cond_compression = is_inside_bar and ((not require_nr4) or is_nr4)
        cond_breakout = price > ib_high
        cond_breakdown = price < ib_low

        confidence = sum([cond_compression, cond_breakout]) / 2.0

        if cond_compression and cond_breakout:
            side = SIDE_BUY
            tags = ["inside bar"]
            if is_nr4:
                tags.append("NR4")
            rationale = (
                f"Crabel compression breakout ({'+'.join(tags)}): yesterday's "
                f"bar nested inside ({ib_low:.2f}–{ib_high:.2f}); current close "
                f"{price:.2f} broke above {ib_high:.2f}. Volatility "
                "expansion expected upward."
            )
        elif cond_compression and cond_breakdown:
            side = SIDE_AVOID
            rationale = (
                f"Compression broke down — close {price:.2f} below inside-bar "
                f"low {ib_low:.2f}. Long-only system avoids the long; the "
                "expansion is going the wrong way."
            )
        else:
            side = SIDE_WAIT
            if not is_inside_bar:
                rationale = (
                    f"No inside bar yet — yesterday's range "
                    f"({ib_low:.2f}–{ib_high:.2f}) not contained inside the "
                    f"prior bar ({ref_low:.2f}–{ref_high:.2f})."
                )
            elif require_nr4 and not is_nr4:
                rationale = (
                    "Inside bar formed but it isn't the narrowest of the last "
                    "4 bars — waiting for tighter compression (NR4)."
                )
            else:
                rationale = (
                    f"Compression in place but no breakout yet — close "
                    f"{price:.2f} still inside ({ib_low:.2f}–{ib_high:.2f})."
                )

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                "inside bar (IB)": is_inside_bar,
                "NR4 (narrowest of last 4)": is_nr4,
                "close > IB high": cond_breakout,
            },
            rationale=rationale,
            extras={
                "ib_high": ib_high,
                "ib_low": ib_low,
                "ib_range": round(ib_range, 4),
            },
            as_of=df.index[-1],
        )
