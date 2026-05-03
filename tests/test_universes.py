from data import universes as U


def test_registry_keys_present() -> None:
    for k in ("us_stocks", "eu_stocks", "asia_stocks", "forex", "commodities", "crypto"):
        assert k in U.UNIVERSES, f"missing universe: {k}"


def test_get_tickers_static_universes_nonempty() -> None:
    for k in ("eu_stocks", "asia_stocks", "forex", "commodities", "crypto"):
        ticks = U.get_tickers(k)
        assert ticks, f"{k} must yield tickers"
        assert all(isinstance(t, str) for t in ticks)


def test_build_selection_caps_us_stocks() -> None:
    sel = U.build_selection(["us_stocks", "forex"], max_per_universe={"us_stocks": 5})
    assert len(sel["us_stocks"]) == 5
    assert len(sel["forex"]) > 5


def test_ticker_to_asset_class_lookup() -> None:
    sel = U.build_selection(["forex", "crypto"])
    lookup = U.ticker_to_asset_class(sel)
    assert lookup["EURUSD=X"] == "Forex"
    assert lookup["BTC-USD"] == "Crypto"


def test_yfinance_ticker_format_sanity() -> None:
    # Forex always carries =X suffix
    for t in U.get_tickers("forex"):
        assert t.endswith("=X"), t
    # Crypto always carries -USD
    for t in U.get_tickers("crypto"):
        assert t.endswith("-USD"), t
    # Asia / EU tickers carry an exchange dot suffix
    for t in U.get_tickers("eu_stocks"):
        assert "." in t, t
