"""Multi-asset universes the dashboard can scan.

Each universe is a curated list of yfinance-compatible tickers. Lists are
intentionally kept reasonably sized — large enough to be useful, small enough
to scan in a few minutes on the free Streamlit Cloud tier.

Universes:
    - us_stocks      S&P 500 (Wikipedia, with hardcoded fallback)
    - eu_stocks      Major European blue-chips (DE/FR/UK/NL/CH/IT/ES)
    - asia_stocks    Major Japan/HK/KR/India/Taiwan large caps
    - forex          Major + popular cross FX pairs (yfinance =X)
    - commodities    Liquid commodity ETFs (gold, silver, oil, agri, copper, …)
    - crypto         Top crypto pairs (yfinance -USD)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# ── S&P 500 fallback (snapshot, not authoritative) ────────────────────────
_SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ORCL",
    "ABBV", "MRK", "BAC", "CVX", "KO", "PEP", "ADBE", "WMT", "CRM", "TMO",
    "MCD", "NFLX", "AMD", "ACN", "CSCO", "LIN", "ABT", "DHR", "WFC", "DIS",
    "VZ", "TXN", "NEE", "PM", "BMY", "INTC", "IBM", "CMCSA", "RTX", "QCOM",
    "PFE", "AMGN", "HON", "UNP", "UPS", "LOW", "T", "INTU", "CAT", "GS",
    "BA", "ELV", "AXP", "DE", "BLK", "SBUX", "GE", "ISRG", "BKNG", "MDT",
    "SPGI", "GILD", "AMT", "TJX", "ADI", "C", "MDLZ", "MMC", "VRTX", "REGN",
    "PLD", "SYK", "LMT", "ETN", "TMUS", "ZTS", "BDX", "CI", "SO", "DUK",
    "PANW", "BSX", "EQIX", "CB", "PGR", "MU", "FI", "AON", "USB", "NOW",
    "CSX", "MO", "SLB", "ITW", "NSC", "CME", "WM", "GD", "MCK", "FDX",
]


@lru_cache(maxsize=1)
def _sp500_live() -> list[str]:
    try:
        tables = pd.read_html(WIKIPEDIA_URL)
        df = tables[0]
        tickers = df["Symbol"].astype(str).tolist()
        tickers = [t.replace(".", "-").strip().upper() for t in tickers if t.strip()]
        if tickers:
            return tickers
    except Exception:
        pass
    seen, out = set(), []
    for t in _SP500_FALLBACK:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── Europe blue chips (yfinance exchange suffixes) ────────────────────────
EU_STOCKS = [
    # 🇩🇪 Germany (.DE — Xetra)
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MBG.DE", "BMW.DE", "VOW3.DE",
    "BAS.DE", "BAYN.DE", "DBK.DE", "MUV2.DE", "ADS.DE", "IFX.DE", "RWE.DE",
    "EOAN.DE", "AIR.DE", "DPW.DE", "HEI.DE", "MTX.DE", "SHL.DE",
    # 🇫🇷 France (.PA)
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA", "ACA.PA",
    "AI.PA", "CS.PA", "DG.PA", "EL.PA", "RMS.PA", "KER.PA", "VIV.PA",
    # 🇬🇧 UK (.L — London)
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "GLEN.L",
    "DGE.L", "BATS.L", "LSEG.L", "REL.L", "VOD.L", "BARC.L", "LLOY.L",
    # 🇳🇱 Netherlands (.AS)
    "ASML.AS", "PRX.AS", "AD.AS", "INGA.AS", "PHIA.AS", "HEIA.AS", "DSM.AS",
    "RAND.AS", "ADYEN.AS",
    # 🇨🇭 Switzerland (.SW)
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "CFR.SW",
    "GIVN.SW", "SREN.SW",
    # 🇮🇹 Italy (.MI)
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STLAM.MI", "RACE.MI",
    # 🇪🇸 Spain (.MC)
    "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC",
]

# ── Asia large caps ───────────────────────────────────────────────────────
ASIA_STOCKS = [
    # 🇯🇵 Japan (.T — Tokyo)
    "7203.T", "6758.T", "9984.T", "8306.T", "6861.T", "7974.T", "9432.T",
    "8058.T", "4063.T", "6501.T", "6098.T", "8316.T", "6981.T", "9433.T",
    "4502.T", "6594.T", "6273.T", "7741.T", "8035.T", "6367.T",
    # 🇭🇰 Hong Kong (.HK)
    "0700.HK", "9988.HK", "0939.HK", "1299.HK", "0388.HK", "0005.HK",
    "1810.HK", "3690.HK", "9618.HK", "2318.HK", "1398.HK", "0883.HK",
    "0941.HK", "2628.HK", "0027.HK",
    # 🇰🇷 South Korea (.KS)
    "005930.KS", "000660.KS", "035420.KS", "207940.KS", "005380.KS",
    "051910.KS", "035720.KS", "068270.KS",
    # 🇮🇳 India (.NS — NSE)
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS",
    # 🇹🇼 Taiwan (.TW)
    "2330.TW", "2317.TW", "2454.TW", "2412.TW",
]

# ── Forex (yfinance =X) ───────────────────────────────────────────────────
FOREX = [
    # Majors
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X",
    # Crosses
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "EURAUD=X", "EURCAD=X", "EURNZD=X",
    "GBPJPY=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X",
    "AUDJPY=X", "AUDNZD=X", "AUDCAD=X", "AUDCHF=X",
    "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
    # Emerging / exotics
    "USDMXN=X", "USDZAR=X", "USDTRY=X", "USDSGD=X", "USDHKD=X", "USDSEK=X",
    "USDNOK=X", "USDCNY=X",
]

# ── Commodities (ETFs preferred for liquidity & continuous data) ──────────
COMMODITIES = [
    # Precious metals
    "GLD",   # Gold
    "SLV",   # Silver
    "PPLT",  # Platinum
    "PALL",  # Palladium
    # Energy
    "USO",   # Crude oil
    "BNO",   # Brent
    "UNG",   # Natural gas
    "UGA",   # Gasoline
    # Industrial metals
    "CPER",  # Copper
    # Agriculture
    "DBA",   # Broad agri
    "CORN",  # Corn
    "WEAT",  # Wheat
    "SOYB",  # Soybeans
    "CANE",  # Sugar
    # Broad baskets
    "DBC",   # Broad commodity index
    "GSG",   # GSCI
    # Direct futures (for comparison — heavier roll noise)
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "ZS=F",
]

# ── Crypto ────────────────────────────────────────────────────────────────
CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD",
    "TRX-USD", "ATOM-USD", "ETC-USD", "XLM-USD", "BCH-USD", "FIL-USD",
    "NEAR-USD", "APT-USD",
]


# ── Registry ──────────────────────────────────────────────────────────────
UNIVERSES: dict[str, dict] = {
    "us_stocks":   {"label": "🇺🇸 US Stocks (S&P 500)", "asset_class": "Stocks US",
                    "fetcher": _sp500_live},
    "eu_stocks":   {"label": "🇪🇺 Europe Stocks",       "asset_class": "Stocks EU",
                    "tickers": EU_STOCKS},
    "asia_stocks": {"label": "🌏 Asia Stocks",           "asset_class": "Stocks Asia",
                    "tickers": ASIA_STOCKS},
    "forex":       {"label": "💱 Forex",                 "asset_class": "Forex",
                    "tickers": FOREX},
    "commodities": {"label": "🛢️ Commodities",          "asset_class": "Commodity",
                    "tickers": COMMODITIES},
    "crypto":      {"label": "₿ Crypto",                "asset_class": "Crypto",
                    "tickers": CRYPTO},
}


def get_tickers(universe_key: str) -> list[str]:
    """Return tickers for one universe key (live fetch where applicable)."""
    spec = UNIVERSES.get(universe_key)
    if spec is None:
        return []
    if "fetcher" in spec:
        return list(spec["fetcher"]())
    return list(spec.get("tickers", []))


def asset_class_for(universe_key: str) -> str:
    return UNIVERSES.get(universe_key, {}).get("asset_class", "Other")


def label_for(universe_key: str) -> str:
    return UNIVERSES.get(universe_key, {}).get("label", universe_key)


def build_selection(keys: list[str], max_per_universe: dict[str, int] | None = None
                    ) -> dict[str, list[str]]:
    """Return ``{universe_key: [tickers]}`` for the selected universes.

    ``max_per_universe`` lets callers cap the size of slow universes (typically
    just the S&P 500) to keep scans fast.
    """
    max_per_universe = max_per_universe or {}
    out: dict[str, list[str]] = {}
    for k in keys:
        ticks = get_tickers(k)
        cap = max_per_universe.get(k)
        if cap:
            ticks = ticks[:cap]
        out[k] = ticks
    return out


def ticker_to_asset_class(selection: dict[str, list[str]]) -> dict[str, str]:
    """Flat lookup ticker → asset class label, useful for tagging signals."""
    lookup: dict[str, str] = {}
    for k, ticks in selection.items():
        ac = asset_class_for(k)
        for t in ticks:
            lookup.setdefault(t, ac)
    return lookup
