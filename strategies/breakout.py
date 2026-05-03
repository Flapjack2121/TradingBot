"""Donchian-style breakout strategy with volume confirmation.

Goes long when:

1. Today's close exceeds the ``donchian_window``-day high (i.e. an N-day high
   breakout — Donchian channel upper-band breach).
2. Today's volume is at least ``volume_multiplier`` × the volume SMA. This
   filters out low-conviction breakouts.
3. Optional regime filter: long-term trend up (close > EMA200).

Verdicts:
- **BUY**   bullish breakout + volume + uptrend,
- **AVOID** active breakdown (close < N-day low), regardless of volume,
- **WAIT**  in range — no breakout, no breakdown.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


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
        prior_low = df["Low"].rolling(window).min().shift(1).iloc[-1]
        last = df.iloc[-1]

        try:
            price = float(last["Close"])
            volume = float(last["Volume"])
            volume_avg = float(last["volume_sma"])
            atr_val = float(last["atr"])
            ema200 = float(last.get("ema200")) if not pd.isna(last.get("ema200")) else None
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)

        if any(pd.isna(x) for x in (prior_high, prior_low, volume, volume_avg, atr_val)):
            return self._empty(ticker, self.name)

        cond_breakout = price > prior_high
        cond_volume = volume >= volume_avg * vol_mult
        cond_trend = (ema200 is None) or (price > ema200)
        cond_breakdown = price < prior_low

        confidence = sum([cond_breakout, cond_volume, cond_trend]) / 3.0
        vol_ratio = (volume / volume_avg) if volume_avg else 0.0

        if cond_breakout and cond_volume and cond_trend:
            side = SIDE_BUY
            rationale = (
                f"Bullish breakout: close {price:.2f} above {window}-day high "
                f"{prior_high:.2f}, volume {vol_ratio:.1f}× average, in uptrend."
            )
        elif cond_breakdown:
            side = SIDE_AVOID
            rationale = (
                f"Bearish breakdown: close {price:.2f} below {window}-day low "
                f"{prior_low:.2f}. Avoid longs — momentum is downward."
            )
        else:
            side = SIDE_WAIT
            missing = []
            if not cond_breakout:
                missing.append(f"price still inside {window}-day range (< {prior_high:.2f})")
            if cond_breakout and not cond_volume:
                missing.append(f"breakout lacks volume confirmation ({vol_ratio:.1f}×, need ≥ {vol_mult}×)")
            if cond_breakout and not cond_trend:
                missing.append("trend filter not bullish (close < EMA200)")
            rationale = "Range-bound — " + ("; ".join(missing) or "waiting for a clean breakout") + "."

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"close > {window}-day high": cond_breakout,
                f"volume ≥ {vol_mult}× SMA": cond_volume,
                "trend (close > EMA200)": cond_trend,
            },
            rationale=rationale,
            extras={
                "donchian_high": float(prior_high),
                "donchian_low": float(prior_low),
                "volume": volume,
                "volume_sma": volume_avg,
                "vol_ratio": round(vol_ratio, 2),
            },
            as_of=df.index[-1],
        )
