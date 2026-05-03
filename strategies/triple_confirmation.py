"""Triple-confirmation pull-back strategy.

A long signal fires when **all three** of the following align on the latest
bar:

1. **Trend** — Close > EMA(200): the symbol is in a long-term uptrend.
2. **Momentum** — RSI(14) crosses 50 from below: fresh bullish energy.
3. **Volatility / value** — Close within ``bb_touch_tolerance`` of the
   lower Bollinger Band: the pull-back is statistically deep.

We emit:
- **BUY**   when all three conditions are true,
- **AVOID** when the trend filter fails (no longs in established downtrends),
- **WAIT**  in every other "uptrend but setup not present" case.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class TripleConfirmation(BaseStrategy):
    name = "triple_confirmation"

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 210:
            return self._empty(ticker, self.name)

        rsi_threshold = self.params.get("rsi_threshold", 50)
        tol = self.params.get("bb_touch_tolerance", 0.01)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        try:
            price = float(last["Close"])
            ema200 = float(last["ema200"])
            rsi_now = float(last["rsi"])
            rsi_prev = float(prev["rsi"])
            bb_lower = float(last["bb_lower"])
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)

        if any(pd.isna(x) for x in (ema200, rsi_now, rsi_prev, bb_lower, atr_val)):
            return self._empty(ticker, self.name)

        cond_trend = price > ema200
        cond_momentum = rsi_prev < rsi_threshold <= rsi_now
        cond_value = price <= bb_lower * (1 + tol)

        confidence = sum([cond_trend, cond_momentum, cond_value]) / 3.0

        if cond_trend and cond_momentum and cond_value:
            side = SIDE_BUY
            rationale = (
                f"All 3 conditions met: uptrend (close {price:.2f} > EMA200 {ema200:.2f}), "
                f"RSI crossed {rsi_threshold} ({rsi_prev:.1f} → {rsi_now:.1f}), "
                f"and price near lower BB ({bb_lower:.2f})."
            )
        elif not cond_trend:
            side = SIDE_AVOID
            rationale = (
                f"Long-term downtrend — close {price:.2f} below EMA200 {ema200:.2f}. "
                "No long setup; wait for trend to repair."
            )
        else:
            missing = []
            if not cond_momentum:
                missing.append(f"momentum (RSI {rsi_now:.1f}, no fresh cross of {rsi_threshold})")
            if not cond_value:
                missing.append(f"value (close {price:.2f} > lower BB {bb_lower:.2f})")
            side = SIDE_WAIT
            rationale = (
                f"Uptrend intact (close {price:.2f} > EMA200 {ema200:.2f}), but waiting for "
                + " and ".join(missing) + "."
            )

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                "trend (close > EMA200)": cond_trend,
                f"momentum (RSI cross {rsi_threshold})": cond_momentum,
                "value (touch lower BB)": cond_value,
            },
            rationale=rationale,
            extras={"rsi": rsi_now, "ema200": ema200, "bb_lower": bb_lower},
            as_of=df.index[-1],
        )
