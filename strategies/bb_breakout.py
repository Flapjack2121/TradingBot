"""Bollinger Band Squeeze + Volatility-Expansion Breakout — universal swing.

Source: John Bollinger, *Bollinger on Bollinger Bands* (2001). The "Squeeze"
setup combined with a directional breakout is one of the most-cited
volatility-expansion systems and works across stocks, FX, commodities and
crypto because it is purely structural — no asset-class assumptions baked in.

Logic on daily bars:

  1.  **Squeeze setup** — bandwidth (BB-Upper − BB-Lower) is at a recent low.
      Default: today's bandwidth ≤ ``squeeze_lookback``-bar bandwidth median
      × ``squeeze_multiple`` (default 0.85).
  2.  **Breakout** — current close exceeds the upper Bollinger Band.
  3.  **Trend filter** — close > EMA(``trend_filter_ema``). Stops us shorting
      ourselves into bear-market false breakouts.

Verdicts:
- **BUY**   squeeze + close above upper band + trend filter passes.
- **AVOID** close below lower band + trend filter fails (volatility expansion
            in the wrong direction).
- **WAIT**  squeezed but no breakout yet, or no squeeze.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, Signal, SIDE_AVOID, SIDE_BUY, SIDE_WAIT


class BollingerBandBreakout(BaseStrategy):
    name = "bb_breakout"
    label = "💥 Bollinger Squeeze Breakout"
    description = (
        "Volatility-squeeze followed by an upper-band breakout, trend-filtered. "
        "Universal across asset classes."
    )
    source = "Bollinger, Bollinger on Bollinger Bands (2001)"
    modes = ["swing"]
    asset_classes = ["all"]

    typical_hold = "Days to ~3 weeks (5–20 bars)"
    typical_hold_bars = (5, 20)
    exit_rules = [
        "Close back inside the upper band on a daily close — failed "
        "breakout, exit at next open.",
        "Price tags the lower Bollinger Band — full mean reversion, take "
        "profits.",
        "Bandwidth re-contracts to a new squeeze — momentum exhausted; "
        "exit and watch for the next setup.",
        "Standard 2R take-profit at +2× initial risk.",
    ]
    watch_for = [
        "Volume on breakout bar — Bollinger himself emphasises volume "
        "confirmation. Low-volume band breaks fail more often.",
        "M-tops: a second band-tag that fails to make a new high → "
        "reversal warning. W-bottoms mirror it on the downside.",
        "Higher timeframe trend — a daily band break against a weekly "
        "downtrend is unreliable. Check weekly chart.",
        "Catalysts: earnings, central bank meetings, OPEC, FOMC — "
        "squeezes often resolve into the news event.",
    ]
    why_it_works = (
        "Markets oscillate between low-volatility (compression) and "
        "high-volatility (expansion) regimes. Bollinger's research, plus "
        "decades of practitioner replication, shows that bandwidth has "
        "predictive power: extended periods of contraction "
        "disproportionately precede expansion events. Trading the "
        "*direction* of the expansion (filtered by trend) catches the "
        "moves where volatility is being re-priced higher."
    )

    def generate(self, ticker: str, df: pd.DataFrame) -> Signal:
        if df is None or df.empty or len(df) < 210:
            return self._empty(ticker, self.name)

        squeeze_lookback = self.params.get("squeeze_lookback", 60)
        squeeze_multiple = self.params.get("squeeze_multiple", 0.85)
        trend_ema = self.params.get("trend_filter_ema", 200)

        last = df.iloc[-1]

        try:
            price = float(last["Close"])
            bb_upper = float(last["bb_upper"])
            bb_lower = float(last["bb_lower"])
            atr_val = float(last["atr"])
            ema_long = float(last.get(f"ema_{trend_ema}",
                                      last.get("ema200", float("nan"))))
        except (KeyError, ValueError, TypeError):
            return self._empty(ticker, self.name)

        if any(pd.isna(x) for x in (bb_upper, bb_lower, atr_val, ema_long)):
            return self._empty(ticker, self.name, "indicator NaN")

        bandwidth_series = (df["bb_upper"] - df["bb_lower"]).dropna()
        if len(bandwidth_series) < squeeze_lookback:
            return self._empty(ticker, self.name, "not enough bandwidth history")

        bw_now = float(bandwidth_series.iloc[-1])
        bw_median = float(bandwidth_series.tail(squeeze_lookback).median())
        cond_squeeze = bw_now <= bw_median * squeeze_multiple

        cond_breakout = price > bb_upper
        cond_breakdown = price < bb_lower
        cond_trend = price > ema_long

        confidence = sum([cond_squeeze, cond_breakout, cond_trend]) / 3.0
        bw_ratio = (bw_now / bw_median) if bw_median else 0.0

        if cond_squeeze and cond_breakout and cond_trend:
            side = SIDE_BUY
            rationale = (
                f"Bollinger squeeze release: bandwidth {bw_ratio:.0%} of "
                f"{squeeze_lookback}-bar median, close {price:.2f} broke upper "
                f"band {bb_upper:.2f}, in uptrend (close > EMA{trend_ema}). "
                "Volatility expansion expected upward."
            )
        elif cond_breakdown and not cond_trend:
            side = SIDE_AVOID
            rationale = (
                f"Lower-band breakdown in downtrend — close {price:.2f} below "
                f"BB-lower {bb_lower:.2f} and EMA{trend_ema} {ema_long:.2f}. "
                "Long-only system avoids; expansion is downward."
            )
        else:
            missing = []
            if not cond_squeeze:
                missing.append(
                    f"no squeeze (bandwidth {bw_ratio:.0%} of median, "
                    f"need ≤ {squeeze_multiple:.0%})"
                )
            if not cond_breakout:
                missing.append(f"close {price:.2f} not above upper band {bb_upper:.2f}")
            if not cond_trend:
                missing.append(f"trend filter (close < EMA{trend_ema})")
            side = SIDE_WAIT
            rationale = "Setup not ready — " + "; ".join(missing) + "."

        return Signal(
            ticker=ticker,
            strategy=self.name,
            side=side,
            entry=price if side == SIDE_BUY else None,
            atr=atr_val,
            confidence=confidence,
            reasons={
                f"squeeze (bandwidth ≤ {squeeze_multiple:.0%} median)": cond_squeeze,
                "close > BB upper": cond_breakout,
                f"trend (close > EMA{trend_ema})": cond_trend,
            },
            rationale=rationale,
            extras={
                "bandwidth": bw_now,
                "bandwidth_median": bw_median,
                "bw_ratio": round(bw_ratio, 3),
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
            },
            as_of=df.index[-1],
        )
