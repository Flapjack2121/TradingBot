from unittest.mock import patch

from data import DataManager


def test_cache_path_scoped_by_interval_and_period(tmp_path) -> None:
    dm = DataManager(cache_dir=tmp_path, period="2y", interval="1d")
    daily = dm._cache_path("AAPL", period="2y", interval="1d")
    hourly = dm._cache_path("AAPL", period="730d", interval="1h")
    assert daily != hourly
    assert "1d_2y" in str(daily)
    assert "1h_730d" in str(hourly)
    # Same call without overrides matches the dataclass defaults
    default = dm._cache_path("AAPL")
    assert "1d_2y" in str(default)


def test_get_passes_interval_to_download(tmp_path, ohlcv) -> None:
    dm = DataManager(cache_dir=tmp_path, period="2y", interval="1d")
    captured: dict = {}

    def fake_download(ticker, period, interval=None):
        captured["ticker"] = ticker
        captured["period"] = period
        captured["interval"] = interval
        return ohlcv

    with patch.object(DataManager, "_download", side_effect=fake_download):
        dm.get("AAPL", period="730d", interval="1h", force_refresh=True)

    assert captured["ticker"] == "AAPL"
    assert captured["period"] == "730d"
    assert captured["interval"] == "1h"


def test_market_regime_uses_daily_spy(tmp_path, ohlcv) -> None:
    """Even in day mode, regime must come from daily SPY."""
    dm = DataManager(cache_dir=tmp_path, period="730d", interval="1h",
                      benchmark="SPY")
    seen: list[tuple] = []

    def fake_download(ticker, period, interval=None):
        seen.append((ticker, period, interval))
        return ohlcv

    with patch.object(DataManager, "_download", side_effect=fake_download):
        dm.market_regime(force_refresh=True)

    assert seen, "expected a download for the benchmark"
    ticker, period, interval = seen[0]
    assert ticker == "SPY"
    assert period == "2y"
    assert interval == "1d"
