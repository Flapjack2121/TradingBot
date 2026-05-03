"""Triple-confirmation pull-back strategy.

A long signal fires when **all three** of the following align on the latest
bar:

1. **Trend** — Close > EMA(200): the symbol is in a long-term uptrend.
2. **Momentum** — RSI(14) crosses 50 from below: fresh bullish energy.
3. **Volatility / value** — Close within ``bb_touch_tolerance`` of the
   lower Bollinger Band: the pull-back is statistically deep.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal


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
        side = "BUY" if cond_trend and cond_momentum and cond_value else "WAIT"

        sig = Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == "BUY" else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                "trend (close > EMA200)": cond_trend,
                f"momentum (RSI cross {rsi_threshold})": cond_momentum,
                "value (touch lower BB)": cond_value,
            },
            extras={"rsi": rsi_now, "ema200": ema200, "bb_lower": bb_lower},
            as_of=df.index[-1],
        )
        return sig
