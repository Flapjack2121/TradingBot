"""Pure pandas/numpy implementations of the indicators we need.

Avoids the pandas-ta / TA-Lib install pain. Functions here all take a single
OHLCV DataFrame with the standard column names ``Open, High, Low, Close, Volume``
and return the same DataFrame with new indicator columns appended.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=window - 1, adjust=False).mean()


def bollinger(close: pd.Series, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + std * sd,
            "bb_lower": mid - std * sd,
            "bb_width": (mid + std * sd - (mid - std * sd)) / mid,
        }
    )


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "donchian_high": high.rolling(window).max(),
            "donchian_low": low.rolling(window).min(),
        }
    )


def compute_indicators(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Append the full indicator stack to a price DataFrame.

    Parameters
    ----------
    df: OHLCV with columns Open/High/Low/Close/Volume.
    cfg: Optional indicator config dict (matches ``indicators`` block of config.yaml).
    """
    cfg = cfg or {}
    out = df.copy()
    if out.empty:
        return out

    close = out["Close"].squeeze()
    high = out["High"].squeeze()
    low = out["Low"].squeeze()
    volume = out["Volume"].squeeze() if "Volume" in out.columns else pd.Series(index=out.index, dtype=float)

    for w in cfg.get("rsi", [2, 14]):
        out[f"rsi_{w}"] = rsi(close, w)
    out["rsi"] = out.get("rsi_14", rsi(close, 14))

    for w in cfg.get("ema", [9, 21, 50, 200]):
        out[f"ema_{w}"] = ema(close, w)
    out["ema200"] = out.get("ema_200")
    out["ema50"] = out.get("ema_50")

    bb = cfg.get("bollinger", {"window": 20, "std": 2})
    out = out.join(bollinger(close, bb["window"], bb["std"]))

    out["atr"] = atr(high, low, close, cfg.get("atr", 14))

    vol_w = cfg.get("volume_sma", 20)
    out[f"volume_sma_{vol_w}"] = volume.rolling(vol_w).mean()
    out["volume_sma"] = out[f"volume_sma_{vol_w}"]

    m = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    out = out.join(macd(close, m["fast"], m["slow"], m["signal"]))

    out = out.join(donchian(high, low, 20))

    out["sma200"] = close.rolling(200).mean()
    return out
