import pandas as pd

from data.indicators import compute_indicators, ema, rsi, atr, bollinger, macd, donchian


def test_ema_matches_pandas(ohlcv: pd.DataFrame) -> None:
    result = ema(ohlcv["Close"], 20)
    expected = ohlcv["Close"].ewm(span=20, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_rsi_in_bounds(ohlcv: pd.DataFrame) -> None:
    r = rsi(ohlcv["Close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_atr_positive(ohlcv: pd.DataFrame) -> None:
    a = atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], 14).dropna()
    assert (a > 0).all()


def test_bollinger_band_order(ohlcv: pd.DataFrame) -> None:
    bb = bollinger(ohlcv["Close"], 20, 2).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all()
    assert (bb["bb_mid"] >= bb["bb_lower"]).all()


def test_macd_columns(ohlcv: pd.DataFrame) -> None:
    m = macd(ohlcv["Close"])
    assert {"macd", "macd_signal", "macd_hist"} <= set(m.columns)


def test_donchian(ohlcv: pd.DataFrame) -> None:
    d = donchian(ohlcv["High"], ohlcv["Low"], 20).dropna()
    assert (d["donchian_high"] >= d["donchian_low"]).all()


def test_compute_indicators_full(ohlcv: pd.DataFrame) -> None:
    out = compute_indicators(ohlcv)
    for col in ("rsi_2", "rsi_14", "ema_50", "ema_200", "bb_upper", "bb_lower",
                "atr", "volume_sma", "macd", "donchian_high", "sma200"):
        assert col in out.columns, f"missing {col}"
    assert len(out) == len(ohlcv)
