"""Session-anchored VWAP Mean Reversion — day-trading institutional standard.

Source: Volume-Weighted Average Price is the institutional benchmark for
intraday execution. Buying a meaningful session-relative dip below VWAP and
selling back toward VWAP is one of the oldest documented intraday edges and
is taught in every prop-firm curriculum.

Logic on intraday bars:

  1.  Compute the session VWAP — anchored at today's first bar:
          VWAP = Σ(typical_price · volume) / Σ(volume)
      where typical_price = (High + Low + Close) / 3.
  2.  Compute the standard deviation of (Close − VWAP) within the session.
  3.  Compute the current bar's z-score: (Close − VWAP) / std.
  4.  Trend filter: today's close above EMA(``trend_filter_ema``) — only fade
      dips in confirmed uptrends.

Verdicts:
- **BUY**   z-score < -``z_threshold`` (deep discount to VWAP) AND uptrend
            — expect mean-reversion bounce back toward VWAP.
- **AVOID** z-score > +``z_threshold`` (extended above VWAP) — buyers are
            paying a premium; sets up a higher-probability reversal/short
            for traders, longs avoid.
- **WAIT**  inside the band, no edge.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class VWAPReversion(BaseStrategy):
    name = "vwap_reversion"
    label = "📊 VWAP Mean Reversion"
    description = (
        "Fade extreme z-score deviations from session VWAP, trend-filtered."
    )
    modes = ["day"]

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 30:
            return self._empty(ticker, self.name)

        z_threshold = self.params.get("z_threshold", 1.5)
        trend_ema = self.params.get("trend_filter_ema", 50)
        min_bars_in_session = self.params.get("min_bars_in_session", 3)

        try:
            today = pd.Timestamp(df.index[-1]).date()
        except Exception:
            return self._empty(ticker, self.name, "bad index")

        today_mask = pd.Series(df.index, index=df.index).apply(
            lambda ts: ts.date() == today
        )
        today_bars = df[today_mask.values]
        if len(today_bars) < min_bars_in_session:
            return Signal(
                ticker=ticker, strategy=self.name, side=SIDE_WAIT,
                rationale=(
                    f"Session too young ({len(today_bars)} bar(s)) for a "
                    "reliable VWAP — need at least "
                    f"{min_bars_in_session}."
                ),
                reasons={"session warmed up": False},
                as_of=df.index[-1],
            )

        typical = (today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3
        cum_vp = (typical * today_bars["Volume"]).cumsum()
        cum_vol = today_bars["Volume"].cumsum().replace(0, pd.NA)
        vwap = (cum_vp / cum_vol).astype(float)
        current_vwap = float(vwap.iloc[-1])

        deviations = today_bars["Close"].astype(float) - vwap
        std = float(deviations.std()) if len(deviations) > 2 else 0.0

        last = df.iloc[-1]
        try:
            price = float(last["Close"])
            atr_val = float(last["atr"])
            ema_long = float(last.get(f"ema_{trend_ema}"))
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name, "missing indicator")
        if pd.isna(ema_long) or pd.isna(atr_val) or std == 0 or pd.isna(current_vwap):
            return self._empty(ticker, self.name, "indicator NaN")

        z = (price - current_vwap) / std

        cond_oversold = z < -z_threshold
        cond_overbought = z > +z_threshold
        cond_trend = price > ema_long

        confidence_buy = sum([cond_oversold, cond_trend]) / 2.0

        if cond_oversold and cond_trend:
            side = SIDE_BUY
            rationale = (
                f"Session VWAP discount: close {price:.2f} is {z:+.2f}σ "
                f"below VWAP {current_vwap:.2f} (threshold {z_threshold:.1f}σ), "
                f"in uptrend (close > EMA{trend_ema} {ema_long:.2f}). "
                "Mean reversion bounce expected toward VWAP."
            )
        elif cond_overbought:
            side = SIDE_AVOID
            rationale = (
                f"Extended above VWAP — close {price:.2f} is {z:+.2f}σ "
                f"above session VWAP {current_vwap:.2f}. Long-only system "
                "avoids; reversion target is back down."
            )
        else:
            side = SIDE_WAIT
            if cond_oversold and not cond_trend:
                rationale = (
                    f"Discount to VWAP ({z:+.2f}σ) but downtrend (close "
                    f"{price:.2f} ≤ EMA{trend_ema} {ema_long:.2f}). Trend "
                    "filter blocks the long."
                )
            else:
                rationale = (
                    f"Inside VWAP band ({z:+.2f}σ, threshold ±{z_threshold:.1f}). "
                    "No edge."
                )
            confidence_buy = sum([cond_oversold, cond_trend]) / 2.0

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence_buy,
            reasons={
                f"z-score < -{z_threshold:.1f}": cond_oversold,
                f"trend (close > EMA{trend_ema})": cond_trend,
                f"z-score > +{z_threshold:.1f} (avoid)": cond_overbought,
            },
            rationale=rationale,
            extras={
                "vwap": current_vwap,
                "z_score": round(z, 3),
                "session_bars": int(len(today_bars)),
            },
            as_of=df.index[-1],
        )
