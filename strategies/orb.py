"""Opening Range Breakout (ORB) — day-trading classic.

Source: Toby Crabel, *Day Trading with Short Term Price Patterns* (1990) and
the broader Market Wizards / institutional intraday literature. Variants are
used by countless prop and futures desks.

Logic on hourly bars:

  1.  Identify the current trading session (today's date in the index).
  2.  Take the high/low of the **first bar(s)** of that session — the
      "opening range" (ORN). With 1h bars, ``opening_bars`` defaults to 1
      (i.e. the first hour).
  3.  Entry: a later bar in the same session closes above the opening-range
      high, with volume confirmation.
  4.  Avoid: the symbol breaks below the opening-range low — opposite
      direction; long-only system stays out.

Verdicts:
- **BUY**   close above opening-range high + volume ≥ multiplier × SMA.
- **AVOID** close below opening-range low (downside ORB — no longs).
- **WAIT**  inside the opening range (still consolidating).
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class OpeningRangeBreakout(BaseStrategy):
    name = "orb"
    label = "🌅 Opening Range Breakout"
    description = "Crabel-style intraday breakout from the session's first bars."
    modes = ["day"]

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 30:
            return self._empty(ticker, self.name)

        opening_bars = self.params.get("opening_bars", 1)
        vol_mult = self.params.get("volume_multiplier", 1.5)

        last = df.iloc[-1]
        try:
            today = pd.Timestamp(df.index[-1]).date()
        except Exception:
            return self._empty(ticker, self.name, "bad index")

        today_mask = pd.Series(df.index, index=df.index).apply(
            lambda ts: ts.date() == today
        )
        today_bars = df[today_mask.values]
        if len(today_bars) < 1:
            return self._empty(ticker, self.name, "no session bars")

        # If today only has one bar, we can't have a post-opening breakout yet.
        if len(today_bars) <= opening_bars:
            return Signal(
                ticker=ticker, strategy=self.name, side=SIDE_WAIT,
                rationale=(
                    f"Session just started — only {len(today_bars)} bar(s) "
                    f"so far, opening range needs {opening_bars}+."
                ),
                reasons={"session warmed up": False},
                as_of=df.index[-1],
            )

        opening = today_bars.iloc[:opening_bars]
        range_high = float(opening["High"].max())
        range_low = float(opening["Low"].min())

        try:
            price = float(last["Close"])
            volume = float(last["Volume"])
            volume_avg = float(last["volume_sma"])
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)
        if any(pd.isna(x) for x in (volume_avg, atr_val)):
            return self._empty(ticker, self.name, "indicator NaN")

        cond_breakout = price > range_high
        cond_breakdown = price < range_low
        cond_volume = volume >= volume_avg * vol_mult
        vol_ratio = (volume / volume_avg) if volume_avg else 0.0

        confidence = sum([cond_breakout, cond_volume]) / 2.0

        if cond_breakout and cond_volume:
            side = SIDE_BUY
            rationale = (
                f"ORB long: close {price:.2f} above opening-range high "
                f"{range_high:.2f} (first {opening_bars} bar(s) of today), "
                f"volume {vol_ratio:.1f}× SMA."
            )
        elif cond_breakdown:
            side = SIDE_AVOID
            rationale = (
                f"Downside ORB — close {price:.2f} below opening-range low "
                f"{range_low:.2f}. Long-only system avoids; the move is "
                "going the wrong way."
            )
        else:
            missing = []
            if not cond_breakout:
                missing.append(f"price still inside opening range "
                               f"({range_low:.2f}–{range_high:.2f})")
            if cond_breakout and not cond_volume:
                missing.append(f"volume too thin ({vol_ratio:.1f}×, need ≥ {vol_mult}×)")
            side = SIDE_WAIT
            rationale = "Consolidating — " + "; ".join(missing) + "."

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"close > opening-range high ({range_high:.2f})": cond_breakout,
                f"volume ≥ {vol_mult}× SMA": cond_volume,
            },
            rationale=rationale,
            extras={
                "range_high": range_high,
                "range_low": range_low,
                "vol_ratio": round(vol_ratio, 2),
                "opening_bars": opening_bars,
            },
            as_of=df.index[-1],
        )
