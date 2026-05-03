"""S&P 500 ticker universe.

We try to fetch the live constituent list from Wikipedia and fall back to a
hardcoded snapshot if the network is unavailable. The hardcoded list is large
enough to be useful but is not exhaustive — it's a safety net.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Snapshot of common S&P 500 names — used only as a fallback if Wikipedia
# scraping fails (offline, rate-limited, etc.). Not authoritative.
_FALLBACK = [
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
    "EOG", "PYPL", "TGT", "EMR", "ICE", "APD", "PSA", "PNC", "AJG", "CL",
    "MAR", "FCX", "NXPI", "MCO", "ROP", "DG", "GM", "OXY", "ADP", "F",
    "TFC", "ECL", "CARR", "AIG", "MNST", "AFL", "HCA", "TRV", "PSX", "MMM",
    "ORLY", "VLO", "MET", "PXD", "ADM", "CHTR", "WELL", "MSI", "AEP", "EW",
    "ROST", "SHW", "BK", "EXC", "PCAR", "STZ", "AZO", "DLR", "OKE", "CTAS",
    "HUM", "DXCM", "KMB", "AMP", "FTNT", "DOW", "SRE", "FIS", "KMI", "TT",
    "CMG", "ALL", "PRU", "LRCX", "ANET", "CTVA", "GIS", "BIIB", "PAYX", "JCI",
    "WMB", "IDXX", "ED", "OTIS", "FAST", "KLAC", "VRSK", "CDNS", "PEG", "RSG",
    "DHI", "SPG", "DLTR", "URI", "NUE", "WBA", "STT", "ON", "EFX", "AME",
    "GWW", "KR", "PPG", "ODFL", "GEHC", "CPRT", "CMI", "HSY", "SNPS", "MTD",
    "MPC", "LEN", "MCHP", "EBAY", "BAX", "GLW", "RMD", "VICI", "FANG", "WST",
    "DD", "TROW", "EIX", "TSCO", "CTSH", "CAH", "ROK", "ZBH", "ALB", "LH",
    "NTAP", "AVB", "WEC", "PCG", "KEYS", "ANSS", "PWR", "WTW", "CDW", "AWK",
    "FE", "EXR", "ULTA", "FTV", "BR", "STE", "MTB", "PHM", "RJF", "VMC",
    "MLM", "FITB", "EQR", "CHD", "DOV", "DRI", "EXPD", "GPN", "HOLX", "RCL",
    "BALL", "CLX", "TER", "TYL", "ETR", "DTE", "ENPH", "TDG", "CCL", "ESS",
    "NDAQ", "PFG", "INVH", "AEE", "GPC", "STX", "BBY", "TRMB", "WAT", "OMC",
    "PPL", "MAA", "WBD", "K", "HUBB", "NTRS", "COO", "HBAN", "RF", "BRO",
    "CFG", "HIG", "VTR", "WY", "STLD", "CINF", "ATO", "FSLR", "WAB", "SBAC",
    "CMS", "STT", "KEY", "MRO", "EXPE", "NRG", "CNC", "ARE", "MOH", "TXT",
    "CTRA", "TSN", "BLDR", "HRL", "LUV", "LDOS", "HPE", "JBHT", "AKAM", "VRSN",
    "EQT", "CNP", "DGX", "DPZ", "TRGP", "NWSA", "FOXA", "NWS", "FOX", "DVA",
    "MGM", "WRK", "CE", "PAYC", "BG", "DOC", "CHRW", "PODD", "BAH", "POOL",
    "AVY", "L", "NVR", "EG", "SWK", "ALGN", "GEN", "FFIV", "JKHY", "ZBRA",
    "EVRG", "WYNN", "INCY", "NDSN", "JNPR", "LNT", "TPR", "SNA", "BBWI", "CAG",
    "PNW", "DAY", "TFX", "IPG", "TECH", "EMN", "CPB", "CRL", "BIO", "VFC",
    "AAL", "NCLH", "FRT", "LKQ", "AOS", "PNR", "AES", "LW", "RVTY", "MKTX",
    "JBL", "BEN", "MOS", "AIZ", "EPAM", "KMX", "GNRC", "QRVO", "MTCH", "WBA",
    "ALLE", "HAS", "NI", "HSIC", "IVZ", "MAS", "BWA", "UHS", "PARA", "DXC",
    "WHR", "FMC", "HII", "TAP", "RHI", "ETSY", "CZR", "PAYC", "FOXA", "NCLH",
]


@lru_cache(maxsize=1)
def get_sp500_tickers(use_network: bool = True) -> list[str]:
    """Return the current S&P 500 ticker symbols (yfinance-formatted)."""
    if use_network:
        try:
            tables = pd.read_html(WIKIPEDIA_URL)
            df = tables[0]
            tickers = df["Symbol"].astype(str).tolist()
            # yfinance uses '-' instead of '.' in tickers like BRK.B
            tickers = [t.replace(".", "-").strip().upper() for t in tickers]
            if tickers:
                return tickers
        except Exception:
            pass
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in _FALLBACK:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
