"""Mark Minervini SEPA / Trend Template — swing momentum filter.

Source: Minervini, *Trade Like a Stock Market Wizard* (2013).

A symbol qualifies as a "Stage 2 advancing" momentum leader when **all eight**
classic Trend-Template criteria align:

  1.  Price > SMA(150) AND Price > SMA(200)
  2.  SMA(150) > SMA(200)
  3.  SMA(200) is trending up for at least one month (~22 bars)
  4.  SMA(50) > SMA(150) AND SMA(50) > SMA(200)
  5.  Price > SMA(50)
  6.  Price is at least 30 % above its 52-week low
  7.  Price is within 25 % of its 52-week high
  8.  Strong relative-strength proxy: 120-bar rate-of-change is positive
      and meaningfully above market noise (default ≥ 10 %)

We emit:
- **BUY**   when all eight conditions are true.
- **AVOID** when conditions 1+2 fail (no Stage-2 trend at all).
- **WAIT**  when the stock is in an uptrend but the full template is not
            yet satisfied — typical for early-Stage-2 setups still building.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class MinerviniTrendTemplate(BaseStrategy):
    name = "minervini"
    label = "🏆 Minervini Trend Template"
    description = (
        "Mark Minervini's 8-criteria SEPA filter for Stage-2 momentum leaders."
    )
    modes = ["swing"]

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 230:
            return self._empty(ticker, self.name)

        roc_period = self.params.get("roc_period", 120)
        rs_threshold = self.params.get("rs_threshold", 0.10)
        low_pct_above = self.params.get("min_pct_above_52w_low", 0.30)
        high_pct_below = self.params.get("max_pct_below_52w_high", 0.25)
        sma200_lookback = self.params.get("sma200_uptrend_bars", 22)

        last = df.iloc[-1]

        try:
            price = float(last["Close"])
            sma50 = float(last["sma_50"])
            sma150 = float(last["sma_150"])
            sma200 = float(last["sma_200"])
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name, "missing indicator")
        if any(pd.isna(x) for x in (sma50, sma150, sma200, atr_val)):
            return self._empty(ticker, self.name, "indicator NaN")

        try:
            sma200_then = float(df["sma_200"].iloc[-sma200_lookback])
        except (IndexError, ValueError):
            sma200_then = float("nan")
        sma200_trending_up = (
            (not pd.isna(sma200_then)) and sma200 > sma200_then
        )

        window_52w = df.tail(252)
        high_52w = float(window_52w["High"].max())
        low_52w = float(window_52w["Low"].min())

        roc = 0.0
        if len(df) > roc_period:
            past = float(df["Close"].iloc[-roc_period - 1])
            if past > 0:
                roc = price / past - 1

        cond1 = price > sma150 and price > sma200
        cond2 = sma150 > sma200
        cond3 = sma200_trending_up
        cond4 = sma50 > sma150 and sma50 > sma200
        cond5 = price > sma50
        cond6 = low_52w > 0 and price >= low_52w * (1 + low_pct_above)
        cond7 = high_52w > 0 and price >= high_52w * (1 - high_pct_below)
        cond8 = roc >= rs_threshold

        all_conds = [cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8]
        confidence = sum(all_conds) / len(all_conds)

        if all(all_conds):
            side = SIDE_BUY
            rationale = (
                f"Full SEPA Trend Template (8/8): close {price:.2f} above "
                f"SMA50/150/200 in proper hierarchy; "
                f"+{(price/low_52w - 1)*100:.0f}% above 52-wk low, "
                f"-{(1 - price/high_52w)*100:.0f}% from 52-wk high; "
                f"{roc_period}-bar ROC {roc*100:+.1f}%."
            )
        elif not (cond1 and cond2):
            side = SIDE_AVOID
            rationale = (
                f"No Stage-2 trend — moving averages out of order "
                f"(price {price:.2f}, SMA150 {sma150:.2f}, SMA200 {sma200:.2f}). "
                "Minervini system stays out."
            )
        else:
            failed = []
            if not cond3: failed.append("SMA200 not yet rising")
            if not cond4: failed.append("SMA50 not above SMA150/200")
            if not cond5: failed.append(f"price below SMA50 ({sma50:.2f})")
            if not cond6: failed.append("price not 30% above 52-wk low")
            if not cond7: failed.append("price still >25% below 52-wk high")
            if not cond8: failed.append(f"weak {roc_period}-bar ROC ({roc*100:+.1f}%)")
            side = SIDE_WAIT
            rationale = (
                f"Stage-2 forming ({sum(all_conds)}/8 criteria met). "
                "Pending: " + ", ".join(failed) + "."
            )

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                "1. price > SMA150 & SMA200": cond1,
                "2. SMA150 > SMA200": cond2,
                "3. SMA200 trending up": cond3,
                "4. SMA50 > SMA150 & SMA200": cond4,
                "5. price > SMA50": cond5,
                f"6. ≥{low_pct_above*100:.0f}% above 52-wk low": cond6,
                f"7. ≤{high_pct_below*100:.0f}% below 52-wk high": cond7,
                f"8. {roc_period}-bar ROC ≥ {rs_threshold*100:.0f}%": cond8,
            },
            rationale=rationale,
            extras={
                "sma_50": sma50, "sma_150": sma150, "sma_200": sma200,
                "high_52w": high_52w, "low_52w": low_52w, "roc": roc,
            },
            as_of=df.index[-1],
        )
