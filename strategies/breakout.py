"""Donchian-style breakout strategy with volume confirmation.

Goes long when:

1. Today's close exceeds the ``donchian_window``-day high (i.e. an N-day high
   breakout — Donchian channel upper-band breach).
2. Today's volume is at least ``volume_multiplier`` × the volume SMA. This
   filters out low-conviction breakouts.
3. Optional regime filter: long-term trend up (close > EMA200) — keeps us out
   of dead-cat-bounce breakouts in bear markets.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal


class DonchianBreakout(BaseStrategy):
    name = "breakout"

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 210:
            return self._empty(ticker, self.name)

        window = self.params.get("donchian_window", 20)
        vol_mult = self.params.get("volume_multiplier", 1.5)

        # Use the donchian high *excluding today* — otherwise today's bar
        # contains itself and the breakout is trivially true.
        prior_high = df["High"].rolling(window).max().shift(1).iloc[-1]
        last = df.iloc[-1]

        try:
            price = float(last["Close"])
            volume = float(last["Volume"])
            volume_avg = float(last["volume_sma"])
            atr_val = float(last["atr"])
            ema200 = float(last.get("ema200")) if not pd.isna(last.get("ema200")) else None
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)

        if any(pd.isna(x) for x in (prior_high, volume, volume_avg, atr_val)):
            return self._empty(ticker, self.name)

        cond_breakout = price > prior_high
        cond_volume = volume >= volume_avg * vol_mult
        cond_trend = (ema200 is None) or (price > ema200)

        confidence = sum([cond_breakout, cond_volume, cond_trend]) / 3.0
        side = "BUY" if cond_breakout and cond_volume and cond_trend else "WAIT"

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == "BUY" else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"close > {window}-day high": cond_breakout,
                f"volume ≥ {vol_mult}× SMA": cond_volume,
                "trend (close > EMA200)": cond_trend,
            },
            extras={"donchian_high": float(prior_high), "volume": volume, "volume_sma": volume_avg},
            as_of=df.index[-1],
        )
