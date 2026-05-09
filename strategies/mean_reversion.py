"""Larry Connors RSI-2 strategy — swing mean reversion.

Source: Connors & Alvarez, *Short Term Trading Strategies That Work* (2008).

The classic Connors recipe:

  1.  Long-only trend filter: Close > SMA(200).
  2.  Setup: RSI(2) < ``rsi_oversold`` (default 5 — the canonical Connors value).
  3.  Optional pullback confirmation: Close < SMA(5) — the symbol has dipped
      below its short-term mean before the entry signal.

Exit (handled by the backtester via R-multiple TP and ATR stop):
The original Connors exit is "RSI(2) > 70" or "Close > SMA(5)". Our generic
backtester uses an R-multiple/ATR exit, which approximates the holding period.

Verdicts:
- **BUY**   trend up + RSI(2) deep oversold (+ pullback if enabled).
- **AVOID** trend down — fading weakness in confirmed downtrends loses
            money over decades of US equity history.
- **WAIT**  trend up but no oversold dip yet.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class ConnorsRSI2(BaseStrategy):
    name = "mean_reversion"
    label = "🔄 Connors RSI-2"
    description = "Larry Connors RSI(2) deep-pullback mean reversion in uptrends."
    modes = ["swing"]

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 210:
            return self._empty(ticker, self.name)

        rsi_period = self.params.get("rsi_period", 2)
        rsi_oversold = self.params.get("rsi_oversold", 5)
        trend_ema = self.params.get("trend_filter_ema", 200)
        require_pullback = self.params.get("require_pullback", True)

        last = df.iloc[-1]
        rsi_col = f"rsi_{rsi_period}"
        ema_col = f"ema_{trend_ema}"

        if rsi_col not in df.columns or ema_col not in df.columns:
            return self._empty(ticker, self.name, "missing indicator")

        try:
            price = float(last["Close"])
            rsi_short = float(last[rsi_col])
            ema_long = float(last[ema_col])
            sma5 = float(last["sma_5"]) if "sma_5" in df.columns else float("nan")
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)

        if any(pd.isna(x) for x in (rsi_short, ema_long, atr_val)):
            return self._empty(ticker, self.name)

        cond_trend = price > ema_long
        cond_oversold = rsi_short < rsi_oversold
        cond_pullback = (not require_pullback) or (
            (not pd.isna(sma5)) and price < sma5
        )

        all_conds = [cond_trend, cond_oversold, cond_pullback]
        confidence = sum(all_conds) / len(all_conds)

        if cond_trend and cond_oversold and cond_pullback:
            side = SIDE_BUY
            rationale = (
                f"Connors RSI-2 setup: uptrend (close {price:.2f} > "
                f"EMA{trend_ema} {ema_long:.2f}), RSI({rsi_period}) deeply "
                f"oversold at {rsi_short:.1f} (< {rsi_oversold})"
                + (f", price below SMA5 ({sma5:.2f})." if require_pullback else ".")
            )
        elif not cond_trend:
            side = SIDE_AVOID
            rationale = (
                f"Downtrend (close {price:.2f} below EMA{trend_ema} "
                f"{ema_long:.2f}). Connors filter forbids longs in confirmed "
                "downtrends — fading weakness here is a falling-knife trap."
            )
        else:
            missing = []
            if not cond_oversold:
                missing.append(
                    f"RSI({rsi_period}) is {rsi_short:.1f}, not yet "
                    f"oversold (< {rsi_oversold})"
                )
            if require_pullback and not cond_pullback:
                missing.append(f"price {price:.2f} above SMA5 {sma5:.2f}")
            side = SIDE_WAIT
            rationale = "Uptrend OK, but " + " and ".join(missing) + "."

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"trend (close > EMA{trend_ema})": cond_trend,
                f"RSI({rsi_period}) < {rsi_oversold}": cond_oversold,
                **({"pullback (price < SMA5)": cond_pullback}
                   if require_pullback else {}),
            },
            rationale=rationale,
            extras={
                f"rsi_{rsi_period}": rsi_short, ema_col: ema_long, "sma_5": sma5,
            },
            as_of=df.index[-1],
        )


# Back-compat alias so old imports keep working
MeanReversion = ConnorsRSI2
