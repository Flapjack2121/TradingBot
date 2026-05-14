"""Floor-trader Pivot Point Breakout — universal day strategy.

Source: classic floor-trader / commodity-pit technique, codified in numerous
texts (e.g. John Person, *A Complete Guide to Technical Trading Tactics*,
2004) and used as institutional intraday levels by every major trading desk.

Logic on intraday bars:

  1.  Compute classic pivot levels from the **prior session's** OHLC:
          P  = (H + L + C) / 3
          R1 = 2P − L         R2 = P + (H − L)
          S1 = 2P − H         S2 = P − (H − L)
  2.  BUY when the current bar closes above R1 with volume confirmation —
      buyers have taken control of the prior day's value area.
  3.  AVOID when the current bar closes below S1 — sellers in control;
      long-only system stays out.

Verdicts:
- **BUY**   close > R1 + volume ≥ multiplier × SMA.
- **AVOID** close < S1 (downside pivot break).
- **WAIT**  close inside the [S1, R1] band — no edge.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class PivotPointBreakout(BaseStrategy):
    name = "pivot_breakout"
    label = "🧭 Pivot Point Breakout"
    description = (
        "Floor-trader classic — break the prior session's R1 / S1 pivots "
        "with volume confirmation."
    )
    source = "Person, A Complete Guide to Technical Trading Tactics (2004)"
    modes = ["day"]
    asset_classes = ["all"]

    typical_hold = "Same day, intraday only (1–6 hourly bars)"
    typical_hold_bars = (1, 6)
    exit_rules = [
        "Take-profit at R2 (next pivot level above R1) — natural target.",
        "Exit if price falls back below the central pivot (P) — breakout "
        "has failed; sellers regained control.",
        "Close all positions before session close.",
        "Tighten stop to entry once R2 is reached and let it run.",
    ]
    watch_for = [
        "Round numbers near pivot levels — psychological levels amplify "
        "self-fulfilling reactions.",
        "Where index futures are vs. their *own* pivots — correlated "
        "moves are higher quality.",
        "Scheduled news within the session — release times can void the "
        "pivot logic entirely.",
        "Volume on the R1 break — institutional algorithms trigger on "
        "level breaks; thin volume = mostly retail = unreliable.",
    ]
    why_it_works = (
        "Pivot points are computed identically by virtually every "
        "professional desk, retail platform, and intraday algorithm in "
        "the world. That universality turns them into self-fulfilling "
        "support/resistance levels — automated stop orders cluster "
        "around them, and trapped traders react predictably when they "
        "break. The R1 → R2 leg has been documented as one of the most "
        "consistent intraday continuation patterns across futures and "
        "FX (1980s floor-trader records and modern intraday studies)."
    )

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 30:
            return self._empty(ticker, self.name)

        vol_mult = self.params.get("volume_multiplier", 1.5)

        try:
            today = pd.Timestamp(df.index[-1]).date()
        except Exception:
            return self._empty(ticker, self.name, "bad index")

        dates = pd.Series(df.index, index=df.index).apply(lambda ts: ts.date())
        prior_dates = sorted({d for d in dates.values if d < today})
        if not prior_dates:
            return self._empty(ticker, self.name, "no prior session")

        prior_session_date = prior_dates[-1]
        prior_bars = df[dates.values == prior_session_date]
        if prior_bars.empty:
            return self._empty(ticker, self.name, "empty prior session")

        prior_high = float(prior_bars["High"].max())
        prior_low = float(prior_bars["Low"].min())
        prior_close = float(prior_bars["Close"].iloc[-1])

        pivot = (prior_high + prior_low + prior_close) / 3.0
        r1 = 2 * pivot - prior_low
        s1 = 2 * pivot - prior_high
        r2 = pivot + (prior_high - prior_low)
        s2 = pivot - (prior_high - prior_low)

        last = df.iloc[-1]
        try:
            price = float(last["Close"])
            volume = float(last["Volume"])
            volume_avg = float(last["volume_sma"])
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)
        if any(pd.isna(x) for x in (volume_avg, atr_val)):
            return self._empty(ticker, self.name, "indicator NaN")

        cond_breakout = price > r1
        cond_breakdown = price < s1
        cond_volume = volume >= volume_avg * vol_mult
        vol_ratio = (volume / volume_avg) if volume_avg else 0.0

        confidence = sum([cond_breakout, cond_volume]) / 2.0

        if cond_breakout and cond_volume:
            side = SIDE_BUY
            rationale = (
                f"Pivot R1 break: close {price:.2f} above prior session R1 "
                f"{r1:.2f} (P {pivot:.2f}), volume {vol_ratio:.1f}× SMA. "
                f"Next target R2 {r2:.2f}."
            )
        elif cond_breakdown:
            side = SIDE_AVOID
            rationale = (
                f"Pivot S1 break: close {price:.2f} below prior session S1 "
                f"{s1:.2f} (P {pivot:.2f}). Sellers in control; next target "
                f"S2 {s2:.2f}. Long-only system stays out."
            )
        else:
            side = SIDE_WAIT
            rationale = (
                f"Inside pivot band (S1 {s1:.2f} – R1 {r1:.2f}, P {pivot:.2f}). "
                "No directional edge."
            )

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"close > R1 ({r1:.2f})": cond_breakout,
                f"volume ≥ {vol_mult}× SMA": cond_volume,
            },
            rationale=rationale,
            extras={
                "pivot": pivot, "R1": r1, "R2": r2, "S1": s1, "S2": s2,
                "vol_ratio": round(vol_ratio, 2),
            },
            as_of=df.index[-1],
        )
