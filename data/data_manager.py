"""Market data orchestration.

The :class:`DataManager` is responsible for:

1. Resolving the universe (S&P 500 by default, or a custom list).
2. Downloading OHLCV history from yfinance.
3. Caching the result to parquet, refreshing only when stale.
4. Computing the full indicator stack on demand.
5. Determining the broad market regime (SPY > SMA200 → bull).
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .indicators import compute_indicators
from .sp500 import get_sp500_tickers

log = logging.getLogger(__name__)


@dataclass
class DataManager:
    cache_dir: Path = Path("data/cache")
    period: str = "2y"
    interval: str = "1d"
    cache_ttl_hours: float = 12.0
    use_adjusted: bool = True
    benchmark: str = "SPY"
    indicator_cfg: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── universe ──────────────────────────────────────────────────────────
    def universe(self, source: str = "sp500", explicit: Iterable[str] | None = None,
                 max_symbols: int | None = None) -> list[str]:
        if explicit:
            tickers = list(explicit)
        elif source == "sp500":
            tickers = get_sp500_tickers()
        else:
            raise ValueError(f"Unknown universe source: {source}")
        if max_symbols:
            tickers = tickers[:max_symbols]
        return tickers

    # ── caching ───────────────────────────────────────────────────────────
    def _cache_path(self, ticker: str, period: str | None = None) -> Path:
        safe = ticker.replace("/", "_").replace("=", "_")
        scope = period or self.period
        sub = self.cache_dir / scope
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{safe}.parquet"

    def _is_fresh(self, path: Path, ttl_hours: float | None = None) -> bool:
        if not path.exists():
            return False
        ttl = ttl_hours if ttl_hours is not None else self.cache_ttl_hours
        age_h = (dt.datetime.now().timestamp() - path.stat().st_mtime) / 3600
        return age_h < ttl

    # ── download ──────────────────────────────────────────────────────────
    def _download(self, ticker: str, period: str | None = None) -> pd.DataFrame:
        df = yf.download(
            ticker,
            period=period or self.period,
            interval=self.interval,
            progress=False,
            auto_adjust=self.use_adjusted,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna(how="all").copy()
        df.index = pd.to_datetime(df.index)
        return df

    def get(self, ticker: str, force_refresh: bool = False,
            period: str | None = None, ttl_hours: float | None = None) -> pd.DataFrame:
        """Return raw OHLCV for one symbol, using cache when fresh.

        ``period`` overrides the default (e.g. ``"10y"`` for backtests). Each
        period gets its own on-disk cache subdirectory so daily-scan data and
        long-history backtest data don't overwrite each other.
        ``ttl_hours`` lets long-history caches live longer than daily ones.
        """
        path = self._cache_path(ticker, period=period)
        if not force_refresh and self._is_fresh(path, ttl_hours=ttl_hours):
            try:
                return pd.read_parquet(path)
            except Exception as exc:
                log.warning("Failed to read cache %s: %s", path, exc)
        df = self._download(ticker, period=period)
        if not df.empty:
            try:
                df.to_parquet(path)
            except Exception as exc:
                log.warning("Failed to write cache %s: %s", path, exc)
        return df

    def get_with_indicators(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        df = self.get(ticker, force_refresh=force_refresh)
        if df.empty:
            return df
        return compute_indicators(df, self.indicator_cfg)

    def bulk(self, tickers: Iterable[str], force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch (and cache) multiple tickers; returns ticker → indicator DataFrame."""
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                df = self.get_with_indicators(t, force_refresh=force_refresh)
                if not df.empty:
                    out[t] = df
            except Exception as exc:
                log.warning("Skipping %s: %s", t, exc)
        return out

    # ── market regime ─────────────────────────────────────────────────────
    def market_regime(self, sma_window: int = 200, force_refresh: bool = False) -> dict:
        df = self.get(self.benchmark, force_refresh=force_refresh)
        if df.empty or len(df) < sma_window:
            return {"regime": "unknown", "spy_close": None, "spy_sma": None}
        close = df["Close"].squeeze()
        sma = close.rolling(sma_window).mean().iloc[-1]
        last = close.iloc[-1]
        regime = "bull" if last > sma else "bear"
        return {
            "regime": regime,
            "spy_close": float(last),
            "spy_sma": float(sma),
            "as_of": df.index[-1].to_pydatetime(),
        }
