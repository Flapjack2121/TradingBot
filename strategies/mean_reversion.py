"""RSI-2 mean-reversion strategy (Larry Connors style).

Signals a long entry on a deeply oversold short-term RSI inside a longer-term
uptrend filter. Specifically:

1. Trend filter: close > EMA(``trend_filter_ema``).
2. Setup: RSI(``rsi_period``) closes below ``rsi_oversold``.

The idea is to fade short-term weakness in established uptrends.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal


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
        side = "BUY" if cond_trend and cond_oversold else "WAIT"

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == "BUY" else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"trend (close > EMA{trend_ema})": cond_trend,
                f"RSI({rsi_period}) < {rsi_oversold}": cond_oversold,
            },
            extras={f"rsi_{rsi_period}": rsi_short, ema_col: ema_long},
            as_of=df.index[-1],
        )
