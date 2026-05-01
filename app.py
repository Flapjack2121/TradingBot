import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-mode CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* DataFrames */
    [data-testid="stDataFrame"] { background-color: #161b22; }
    /* Section headers */
    h1, h2, h3 { color: #58a6ff !important; }
    /* Divider */
    hr { border-color: #30363d; }
    /* Selectbox / slider labels */
    label { color: #8b949e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "EURUSD=X", "GBPUSD=X", "BTC-USD"]
ACCOUNT_SIZE = 10_000          # € – used for position sizing
RISK_PCT = 0.01                # risk 1 % of account per trade
ATR_MULTIPLIER = 1.5           # stop-loss = entry - ATR * multiplier
REFRESH_INTERVAL = 60          # seconds

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    account_size = st.number_input(
        "Account Size (€)", min_value=100, max_value=1_000_000,
        value=ACCOUNT_SIZE, step=500,
    )
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.25) / 100
    atr_mult = st.slider("ATR Stop-Loss Multiplier", 1.0, 3.0, 1.5, 0.25)
    interval = st.selectbox("Chart Interval", ["1d", "1h", "4h", "15m", "5m"], index=0)
    period = st.selectbox("History Period", ["3mo", "6mo", "1y", "2y"], index=1)

    st.markdown("---")
    st.markdown("### 📋 Watchlist")
    raw = st.text_area(
        "Tickers (one per line)",
        value="\n".join(DEFAULT_WATCHLIST),
        height=200,
    )
    watchlist = [t.strip().upper() for t in raw.splitlines() if t.strip()]

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-Refresh every 60 s", value=True)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    df["ema200"] = _ema(close, 200)
    df["ema50"] = _ema(close, 50)
    df["rsi"] = _rsi(close, 14)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_mid"] = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    df["atr"] = _atr(high, low, close, 14)

    return df


def triple_confirmation(df: pd.DataFrame) -> dict:
    """Returns signal info for the latest candle."""
    if df.empty or len(df) < 210:
        return {"signal": "INSUFFICIENT DATA", "details": {}}

    row = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(row["Close"])
    ema200 = float(row["ema200"]) if not pd.isna(row.get("ema200", float("nan"))) else None
    rsi_now = float(row["rsi"]) if not pd.isna(row.get("rsi", float("nan"))) else None
    rsi_prev = float(prev["rsi"]) if not pd.isna(prev.get("rsi", float("nan"))) else None
    bb_lower = float(row["bb_lower"]) if not pd.isna(row.get("bb_lower", float("nan"))) else None
    atr_val = float(row["atr"]) if not pd.isna(row.get("atr", float("nan"))) else None

    cond1 = (ema200 is not None) and (price > ema200)
    cond2 = (rsi_now is not None and rsi_prev is not None) and (rsi_prev < 50 <= rsi_now)
    cond3 = (bb_lower is not None) and (price <= bb_lower * 1.01)

    signal = "BUY" if (cond1 and cond2 and cond3) else "WAIT"

    return {
        "signal": signal,
        "price": price,
        "ema200": ema200,
        "rsi": rsi_now,
        "bb_lower": bb_lower,
        "atr": atr_val,
        "cond_trend": cond1,
        "cond_momentum": cond2,
        "cond_bb": cond3,
    }


def position_sizing(price: float, atr: float, account: float, risk: float, mult: float) -> dict:
    """Kelly-free fixed-fractional position sizing."""
    stop_loss_dist = atr * mult
    stop_loss_price = price - stop_loss_dist
    risk_amount = account * risk
    # shares/units
    units = risk_amount / stop_loss_dist if stop_loss_dist > 0 else 0
    position_value = units * price
    return {
        "units": round(units, 4),
        "stop_loss": round(stop_loss_price, 4),
        "risk_amount": round(risk_amount, 2),
        "position_value": round(position_value, 2),
    }


# ── Build signal table ────────────────────────────────────────────────────────

def build_signal_table(watchlist, period, interval, account, risk, atr_mult):
    rows = []
    for ticker in watchlist:
        df = fetch(ticker, period, interval)
        if df.empty:
            rows.append({
                "Ticker": ticker, "Price": "—", "Signal": "NO DATA",
                "Trend ✓": False, "RSI Cross ✓": False, "BB Touch ✓": False,
                "RSI": "—", "Stop-Loss": "—", "Units": "—", "Position €": "—",
            })
            continue

        df = compute_indicators(df)
        info = triple_confirmation(df)

        sig = info.get("signal", "—")
        price = info.get("price")
        atr = info.get("atr")

        pos = {}
        if price and atr:
            pos = position_sizing(price, atr, account, risk, atr_mult)

        rows.append({
            "Ticker": ticker,
            "Price": f"{price:.4f}" if price else "—",
            "Signal": sig,
            "Trend ✓": info.get("cond_trend", False),
            "RSI Cross ✓": info.get("cond_momentum", False),
            "BB Touch ✓": info.get("cond_bb", False),
            "RSI": f"{info['rsi']:.1f}" if info.get("rsi") else "—",
            "Stop-Loss": f"{pos.get('stop_loss', '—')}",
            "Units": f"{pos.get('units', '—')}",
            "Position €": f"{pos.get('position_value', '—')}",
        })
    return pd.DataFrame(rows)


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = compute_indicators(df)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} – Candlestick", "Volume", "RSI (14)"),
    )

    # ── Candlesticks ──
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
        ),
        row=1, col=1,
    )

    # EMAs
    for col, color, name in [
        ("ema50", "#d29922", "EMA 50"),
        ("ema200", "#58a6ff", "EMA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name,
                           line=dict(color=color, width=1.5), opacity=0.85),
                row=1, col=1,
            )

    # Bollinger Bands
    if "bb_upper" in df.columns:
        for col, name, dash in [
            ("bb_upper", "BB Upper", "dot"),
            ("bb_mid",   "BB Mid",   "dash"),
            ("bb_lower", "BB Lower", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name,
                           line=dict(color="#8b949e", width=1, dash=dash), opacity=0.6),
                row=1, col=1,
            )
        # shaded band
        fig.add_trace(
            go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(df["bb_upper"]) + list(df["bb_lower"][::-1]),
                fill="toself", fillcolor="rgba(139,148,158,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, name="BB Band",
            ),
            row=1, col=1,
        )

    # ── Volume ──
    colors = [
        "#3fb950" if c >= o else "#f85149"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    # ── RSI ──
    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                       line=dict(color="#bc8cff", width=1.5)),
            row=3, col=1,
        )
        for level, color in [(70, "#f85149"), (50, "#8b949e"), (30, "#3fb950")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#e6edf3"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                    font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=680,
    )
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)
    return fig


# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown("# 📈 Trading Signal Dashboard")
st.caption(
    "Triple-Confirmation Strategy  •  No-API  •  Human-in-the-Loop  •  "
    f"Account: **{account_size:,.0f} €**  |  Risk/Trade: **{risk_pct*100:.2f} %**"
)
st.markdown("---")

# ── Signal table ──────────────────────────────────────────────────────────────
st.markdown("## 🚦 Signal Overview")

with st.spinner("Fetching market data…"):
    sig_df = build_signal_table(watchlist, period, interval, account_size, risk_pct, atr_mult)

def _style_signal(val):
    if val == "BUY":
        return "background-color: #1a3a2a; color: #3fb950; font-weight: bold; border-radius: 4px; padding: 2px 8px;"
    if val == "WAIT":
        return "background-color: #3a1a1a; color: #f85149; font-weight: bold; border-radius: 4px; padding: 2px 8px;"
    return "color: #8b949e;"

def _style_bool(val):
    if val is True:
        return "color: #3fb950;"
    if val is False:
        return "color: #f85149;"
    return ""

styled = (
    sig_df.style
    .map(_style_signal, subset=["Signal"])
    .map(_style_bool, subset=["Trend ✓", "RSI Cross ✓", "BB Touch ✓"])
    .set_properties(**{"background-color": "#161b22", "color": "#e6edf3",
                        "border": "1px solid #30363d"})
    .set_table_styles([
        {"selector": "th", "props": [("background-color", "#21262d"),
                                      ("color", "#58a6ff"),
                                      ("border", "1px solid #30363d")]},
    ])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# Legend
st.markdown(
    """
    <small>
    🟢 <b>Trend ✓</b>: Price > EMA 200 &nbsp;|&nbsp;
    🟢 <b>RSI Cross ✓</b>: RSI crossed 50 upward &nbsp;|&nbsp;
    🟢 <b>BB Touch ✓</b>: Price touched lower Bollinger Band
    </small>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Chart section ─────────────────────────────────────────────────────────────
st.markdown("## 🕯️ Interactive Chart")

col_left, col_right = st.columns([2, 1])

with col_left:
    selected = st.selectbox("Select Ticker", watchlist, key="chart_ticker")

with col_right:
    st.markdown("<br>", unsafe_allow_html=True)
    row = sig_df[sig_df["Ticker"] == selected]
    if not row.empty:
        sig_val = row.iloc[0]["Signal"]
        color = "#3fb950" if sig_val == "BUY" else "#f85149"
        st.markdown(
            f'<div style="background:{color}22;border:1px solid {color};border-radius:8px;'
            f'padding:12px 20px;text-align:center;font-size:1.4rem;font-weight:bold;color:{color};">'
            f'{"🟢" if sig_val == "BUY" else "🔴"} {sig_val}</div>',
            unsafe_allow_html=True,
        )

with st.spinner(f"Loading chart for {selected}…"):
    chart_df = fetch(selected, period, interval)

if not chart_df.empty:
    fig = build_chart(chart_df, selected)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"No data available for {selected}.")

# ── Risk detail card ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Risk Management Detail")

detail_row = sig_df[sig_df["Ticker"] == selected]
if not detail_row.empty:
    dr = detail_row.iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Price", dr["Price"])
    m2.metric("RSI (14)", dr["RSI"])
    m3.metric("Stop-Loss Price", dr["Stop-Loss"])
    m4.metric("Units to Buy", dr["Units"])
    m5.metric("Position Value (€)", dr["Position €"])

st.markdown(
    """
    <small style="color:#8b949e;">
    ⚠️ <b>Disclaimer:</b> This dashboard is for educational purposes only.
    No orders are executed automatically. Always do your own research before trading.
    Stop-loss is calculated as <i>Entry – ATR × Multiplier</i>.
    Position size is based on fixed-fractional risk (account × risk %).
    </small>
    """,
    unsafe_allow_html=True,
)

# ── Strategy explanation ──────────────────────────────────────────────────────
with st.expander("ℹ️ How the Triple-Confirmation Strategy works"):
    st.markdown(
        """
        | # | Condition | Indicator | Why |
        |---|-----------|-----------|-----|
        | 1 | **Trend** | Price > EMA 200 | Confirms the asset is in a long-term uptrend |
        | 2 | **Momentum** | RSI crosses 50 upward | Fresh bullish momentum is building |
        | 3 | **Volatility** | Price touches Lower Bollinger Band | Price is at a statistically cheap level |

        A **BUY** signal fires only when **all three** conditions align simultaneously.
        This filters out noise and focuses on high-probability setups.

        **Risk Management formula:**
        ```
        Stop-Loss   = Current Price  −  (ATR × multiplier)
        Risk Amount = Account Size   ×  Risk %
        Units       = Risk Amount    ÷  Stop-Loss Distance
        ```
        """
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
