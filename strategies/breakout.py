"""Turtle Trading System 1 — swing breakout.

Source: Richard Dennis & William Eckhardt, the Turtle Trading System (1983).
Documented by Curtis Faith in *Way of the Turtle* (2007).

The original Turtle System 1 (S1) rules adapted for long-only equities:

  1.  Entry: today's close breaks above the prior ``donchian_window``-bar high
      (default 20). Optional volume confirmation requires today's volume to
      exceed ``volume_multiplier`` × volume SMA — Dennis didn't require this
      but it filters fake breakouts on equities.
  2.  Optional regime filter: only take longs when the symbol's long-term
      trend is up (close > EMA200) — keeps the system out of failed
      breakouts in confirmed downtrends.
  3.  Stop loss: 2N below entry, where N = ATR(14). Position sizing in the
      original system is 1 % equity ÷ N; we use the engine's ATR-multiple
      stop with the user-configured risk %.

Verdicts:
- **BUY**   N-bar high breakout with volume + uptrend.
- **AVOID** active N-bar low breakdown — opposite of breakout, exit / no
            new longs.
- **WAIT**  inside the prior range — no signal.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class TurtleSystem(BaseStrategy):
    name = "breakout"
    label = "🐢 Turtle Donchian (S1)"
    description = (
        "Richard Dennis Turtle System 1 — N-bar Donchian breakout with "
        "volume confirmation."
    )
    modes = ["swing"]

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 210:
            return self._empty(ticker, self.name)

        window = self.params.get("donchian_window", 20)
        vol_mult = self.params.get("volume_multiplier", 1.5)

        # Donchian channel computed *excluding* today's bar — otherwise the
        # current bar contains itself and the breakout is trivially true.
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
                f"Turtle S1 breakout: close {price:.2f} above {window}-bar "
                f"high {prior_high:.2f}, volume {vol_ratio:.1f}× SMA, in "
                "uptrend. Stop = 2N (2× ATR) below entry per Turtle rules."
            )
        elif cond_breakdown:
            side = SIDE_AVOID
            rationale = (
                f"Turtle short signal — close {price:.2f} below {window}-bar "
                f"low {prior_low:.2f}. Long-only system stays out; existing "
                "longs would have stopped out."
            )
        else:
            missing = []
            if not cond_breakout:
                missing.append(f"price still inside {window}-bar range (< {prior_high:.2f})")
            if cond_breakout and not cond_volume:
                missing.append(
                    f"breakout lacks volume ({vol_ratio:.1f}×, need ≥ {vol_mult}×)"
                )
            if cond_breakout and not cond_trend:
                missing.append("trend filter not bullish (close < EMA200)")
            side = SIDE_WAIT
            rationale = (
                "No breakout — "
                + ("; ".join(missing) or "waiting for a clean N-bar high break")
                + "."
            )

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"close > {window}-bar high": cond_breakout,
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


# Back-compat alias
DonchianBreakout = TurtleSystem
