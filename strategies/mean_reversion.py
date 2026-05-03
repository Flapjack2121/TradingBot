"""RSI-2 mean-reversion strategy (Larry Connors style).

Signals a long entry on a deeply oversold short-term RSI inside a longer-term
uptrend filter. Specifically:

1. Trend filter: close > EMA(``trend_filter_ema``).
2. Setup: RSI(``rsi_period``) closes below ``rsi_oversold``.

Verdicts:
- **BUY**   trend up + RSI(2) deeply oversold,
- **AVOID** trend down — fading weakness in downtrends is a falling-knife trap,
- **WAIT**  trend up but no oversold dip yet.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class MeanReversion(BaseStrategy):
    name = "mean_reversion"

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 210:
            return self._empty(ticker, self.name)

        rsi_period = self.params.get("rsi_period", 2)
        rsi_oversold = self.params.get("rsi_oversold", 10)
        trend_ema = self.params.get("trend_filter_ema", 200)

        last = df.iloc[-1]
        rsi_col = f"rsi_{rsi_period}"
        ema_col = f"ema_{trend_ema}"

        if rsi_col not in df.columns or ema_col not in df.columns:
            return self._empty(ticker, self.name, "missing indicator")

        try:
            price = float(last["Close"])
            rsi_short = float(last[rsi_col])
            ema_long = float(last[ema_col])
            atr_val = float(last["atr"])
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)

        if any(pd.isna(x) for x in (rsi_short, ema_long, atr_val)):
            return self._empty(ticker, self.name)

        cond_trend = price > ema_long
        cond_oversold = rsi_short < rsi_oversold

        confidence = sum([cond_trend, cond_oversold]) / 2.0

        if cond_trend and cond_oversold:
            side = SIDE_BUY
            rationale = (
                f"Setup: uptrend (close {price:.2f} > EMA{trend_ema} {ema_long:.2f}) and "
                f"RSI({rsi_period}) deeply oversold at {rsi_short:.1f} (< {rsi_oversold}). "
                "Mean reversion bounce expected."
            )
        elif not cond_trend:
            side = SIDE_AVOID
            rationale = (
                f"Downtrend (close {price:.2f} below EMA{trend_ema} {ema_long:.2f}) — "
                "no mean-reversion long; fading weakness here is a falling-knife trap."
            )
        else:
            side = SIDE_WAIT
            rationale = (
                f"Uptrend OK but RSI({rsi_period}) is {rsi_short:.1f}, not yet oversold "
                f"(< {rsi_oversold}). Wait for a sharp dip."
            )

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
            },
            rationale=rationale,
            extras={f"rsi_{rsi_period}": rsi_short, ema_col: ema_long},
            as_of=df.index[-1],
        )
