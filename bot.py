"""Trading Signal Dashboard — main Streamlit application.

Run with:

    streamlit run bot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run bot.py`.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings
from data import DataManager
from strategies import REGISTRY, SignalEngine
from strategies.engine import RiskConfig
from database import TradeRepo, TradeStatus
from reports import Reporter
from notifications import Notifier
from notifications.notifier import EmailConfig, TelegramConfig

from dashboard.components import (
    build_chart, render_signal_table, render_stats, render_trade_card,
)


# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS once.
css_path = Path(__file__).parent / "dashboard" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Resource construction (cached) ───────────────────────────────────────
@st.cache_resource
def get_data_manager() -> DataManager:
    cfg = settings.load_config()
    data_cfg = cfg.get("data", {})
    universe_cfg = cfg.get("universe", {})
    return DataManager(
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        period=data_cfg.get("history_period", "2y"),
        interval=data_cfg.get("interval", "1d"),
        cache_ttl_hours=data_cfg.get("cache_ttl_hours", 12),
        use_adjusted=data_cfg.get("use_adjusted", True),
        benchmark=universe_cfg.get("benchmark", "SPY"),
        indicator_cfg=cfg.get("indicators", {}),
    )


@st.cache_resource
def get_repo() -> TradeRepo:
    cfg = settings.load_config()
    return TradeRepo(cfg.get("database", {}).get("path", "database/trades.db"))


def get_engine(account: float, risk_pct: float, atr_mult: float, tp_r: float,
               enabled: list[str], strat_params: dict) -> SignalEngine:
    risk = RiskConfig(
        account_size=account,
        risk_per_trade=risk_pct,
        atr_stop_multiplier=atr_mult,
        take_profit_r_multiple=tp_r,
    )
    strategies = []
    for name in enabled:
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        strategies.append(cls(strat_params.get(name, {})))
    return SignalEngine(strategies=strategies, risk=risk)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_universe(source: str, explicit: tuple[str, ...], max_symbols: int) -> list[str]:
    dm = get_data_manager()
    return dm.universe(source=source, explicit=list(explicit) if explicit else None,
                       max_symbols=max_symbols)


@st.cache_data(ttl=900, show_spinner=False)
def scan_universe(tickers: tuple[str, ...], force: bool, account: float, risk_pct: float,
                  atr_mult: float, tp_r: float, enabled: tuple[str, ...]) -> dict:
    """Run the full pipeline and cache results for 15 minutes."""
    dm = get_data_manager()
    cfg = settings.load_config()
    strat_params = cfg.get("strategies", {})
    engine = get_engine(account, risk_pct, atr_mult, tp_r, list(enabled), strat_params)

    frames = dm.bulk(tickers, force_refresh=force)
    signals = engine.scan(frames)
    regime = dm.market_regime(sma_window=cfg.get("regime", {}).get("spy_sma", 200),
                              force_refresh=force)
    return {"frames": frames, "signals": signals, "regime": regime}


# ── Sidebar controls ─────────────────────────────────────────────────────
cfg = settings.load_config()
risk_cfg = cfg.get("risk", {})
universe_cfg = cfg.get("universe", {})
strategy_cfg = cfg.get("strategies", {})

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    account_size = st.number_input(
        "Account Size (€)", min_value=100, max_value=10_000_000,
        value=int(risk_cfg.get("account_size", 10000)), step=500,
    )
    risk_pct = st.slider(
        "Risk per Trade (%)", 0.1, 5.0,
        float(risk_cfg.get("risk_per_trade", 0.01)) * 100, 0.1,
    ) / 100
    atr_mult = st.slider(
        "ATR Stop Multiplier", 0.5, 5.0,
        float(risk_cfg.get("atr_stop_multiplier", 2.0)), 0.25,
    )
    tp_r = st.slider(
        "Take Profit (R)", 0.5, 5.0,
        float(risk_cfg.get("take_profit_r_multiple", 2.0)), 0.25,
    )

    st.markdown("---")
    st.markdown("### 🎯 Universe")
    universe_mode = st.radio(
        "Source", ["S&P 500", "Custom watchlist"],
        index=0 if universe_cfg.get("source", "sp500") == "sp500" else 1,
    )
    if universe_mode == "S&P 500":
        max_syms = st.slider(
            "Max symbols to scan", 10, 503,
            int(universe_cfg.get("max_symbols", 100)), 10,
        )
        custom: tuple[str, ...] = ()
        source = "sp500"
    else:
        raw = st.text_area(
            "Tickers (one per line)",
            value="\n".join(universe_cfg.get("tickers", []) or ["AAPL", "MSFT", "NVDA", "TSLA"]),
            height=180,
        )
        custom = tuple(t.strip().upper() for t in raw.splitlines() if t.strip())
        max_syms = len(custom)
        source = "custom"

    st.markdown("---")
    st.markdown("### 🧠 Strategies")
    enabled_default = strategy_cfg.get("enabled", list(REGISTRY.keys()))
    enabled = []
    for name in REGISTRY.keys():
        if st.checkbox(name.replace("_", " ").title(), value=name in enabled_default,
                       key=f"strat_{name}"):
            enabled.append(name)
    enabled_t = tuple(enabled)

    st.markdown("---")
    force = st.checkbox("Force refresh data", value=False)
    if st.button("🔄 Re-scan now"):
        st.cache_data.clear()
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────
st.markdown(f"# 📈 {cfg.get('dashboard', {}).get('title', 'Trading Signal Dashboard')}")
st.caption(
    f"Account: **€{account_size:,.0f}**  •  "
    f"Risk/Trade: **{risk_pct*100:.2f}%**  •  "
    f"Stop: **{atr_mult}× ATR**  •  TP: **{tp_r}R**  •  "
    "Human-in-the-loop — no automated execution."
)
st.markdown("---")

if source == "sp500":
    tickers = tuple(fetch_universe("sp500", (), max_syms))
else:
    tickers = custom

if not tickers:
    st.warning("No tickers selected. Add a watchlist or pick S&P 500 in the sidebar.")
    st.stop()

if not enabled:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

with st.spinner(f"Scanning {len(tickers)} symbols across {len(enabled)} strategies…"):
    result = scan_universe(tickers, force, account_size, risk_pct, atr_mult, tp_r, enabled_t)

regime = result["regime"]
frames: dict[str, pd.DataFrame] = result["frames"]
signals = result["signals"]

# ── Market regime banner ────────────────────────────────────────────────
regime_label = regime.get("regime", "unknown")
spy_close = regime.get("spy_close")
spy_sma = regime.get("spy_sma")
banner_class = "regime-bull" if regime_label == "bull" else ("regime-bear" if regime_label == "bear" else "")
banner_emoji = {"bull": "🐂", "bear": "🐻"}.get(regime_label, "❓")
st.markdown(
    f"### Market regime: <span class='{banner_class}'>{banner_emoji} {regime_label.upper()}</span>"
    + (f"  &nbsp;<small>SPY {spy_close:.2f} vs SMA200 {spy_sma:.2f}</small>"
       if spy_close and spy_sma else ""),
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Signal table ────────────────────────────────────────────────────────
engine = get_engine(account_size, risk_pct, atr_mult, tp_r, enabled, strategy_cfg)
df_all = engine.to_dataframe(signals)
actionable = engine.actionable(signals)

tab_signals, tab_actionable, tab_chart, tab_journal, tab_backtest = st.tabs(
    ["🚦 All Signals", "🎯 Actionable", "🕯️ Chart", "📒 Journal", "🧪 Backtest"]
)

with tab_signals:
    st.markdown("### Signal Overview")
    if not df_all.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbols scanned", len(frames))
        c2.metric("Strategies", len(enabled))
        c3.metric("BUY signals", int((df_all["Signal"] == "BUY").sum()))
        c4.metric("WAIT", int((df_all["Signal"] == "WAIT").sum()))
    only_buy = st.checkbox("Show only BUY signals", value=False)
    view = df_all[df_all["Signal"] == "BUY"] if only_buy else df_all
    render_signal_table(view)

with tab_actionable:
    st.markdown(f"### Actionable trades ({len(actionable)})")
    if not actionable:
        st.info("No actionable signals on this scan.")
    repo = get_repo()
    for s in sorted(actionable, key=lambda x: -x.confidence):
        render_trade_card(s, repo, key_prefix="act")

with tab_chart:
    st.markdown("### Interactive chart")
    selectable = sorted(frames.keys())
    if not selectable:
        st.info("No data fetched.")
    else:
        sel = st.selectbox("Ticker", selectable, key="chart_ticker")
        df_sel = frames[sel]
        sigs_sel = [s for s in signals if s.ticker == sel]
        c1, c2 = st.columns([2, 1])
        with c2:
            for s in sigs_sel:
                badge_color = "#3fb950" if s.is_actionable else "#8b949e"
                st.markdown(
                    f"<div style='border:1px solid {badge_color};border-radius:8px;"
                    f"padding:8px 12px;margin-bottom:6px;'>"
                    f"<b style='color:{badge_color};'>{s.strategy}</b>: {s.side} "
                    f"({s.confidence:.0%})</div>",
                    unsafe_allow_html=True,
                )
        with c1:
            best = next((s for s in sigs_sel if s.is_actionable), sigs_sel[0] if sigs_sel else None)
            st.plotly_chart(build_chart(df_sel, sel, signal=best), use_container_width=True)

with tab_journal:
    st.markdown("### Trade journal")
    repo = get_repo()
    render_stats(repo.stats())
    df_trades = repo.to_dataframe()
    if df_trades.empty:
        st.info("No trades recorded yet.")
    else:
        st.dataframe(df_trades, use_container_width=True, hide_index=True)

        st.markdown("#### Close an open trade")
        opens = repo.open_trades()
        if not opens:
            st.caption("No open trades to close.")
        else:
            options = {f"#{t.id} {t.ticker} @ {t.entry}": t.id for t in opens}
            choice = st.selectbox("Open trade", list(options.keys()))
            exit_price = st.number_input("Exit price", min_value=0.0, value=0.0, step=0.01)
            if st.button("Close trade") and exit_price > 0:
                t = repo.close_trade(options[choice], exit_price)
                if t:
                    st.success(f"Closed #{t.id}: PnL €{t.pnl} ({t.realized_r}R).")
                    st.rerun()

        st.markdown("#### Export")
        col_e1, col_e2 = st.columns(2)
        if col_e1.button("📤 Export trades to CSV"):
            path = Reporter().save_trades(df_trades, fmt="csv")
            st.success(f"Saved {path}")
        if col_e2.button("📤 Export signals to JSON"):
            path = Reporter().save_signals(signals, fmt="json")
            st.success(f"Saved {path}")

with tab_backtest:
    st.markdown("### Single-symbol backtest")
    if not frames:
        st.info("No data loaded.")
    else:
        from backtest import Backtester
        bt_cfg = cfg.get("backtest", {})
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            bt_ticker = st.selectbox("Ticker", sorted(frames.keys()), key="bt_ticker")
        with col_b:
            bt_strat_name = st.selectbox("Strategy", enabled, key="bt_strat")
        with col_c:
            bt_cash = st.number_input("Initial cash €", min_value=100, value=int(bt_cfg.get("initial_cash", 10000)))

        if st.button("Run backtest"):
            with st.spinner("Backtesting…"):
                strat_cls = REGISTRY[bt_strat_name]
                strat = strat_cls(strategy_cfg.get(bt_strat_name, {}))
                bt = Backtester(
                    initial_cash=bt_cash,
                    risk_per_trade=risk_pct,
                    atr_stop_multiplier=atr_mult,
                    take_profit_r_multiple=tp_r,
                    commission_pct=bt_cfg.get("commission_pct", 0.0005),
                    slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
                )
                bt_res = bt.run(bt_ticker, frames[bt_ticker], strat)
            st.json(bt_res.stats)
            if not bt_res.equity_curve.empty:
                st.line_chart(bt_res.equity_curve, height=320)
            if not bt_res.trades.empty:
                st.markdown("##### Trades")
                st.dataframe(bt_res.trades, use_container_width=True, hide_index=True)


# ── Notifications (manual trigger) ───────────────────────────────────────
st.markdown("---")
with st.expander("📨 Notifications"):
    notif_cfg = cfg.get("notifications", {})
    st.caption(
        "Configure in `config.yaml`. Channels enabled: "
        + ", ".join(notif_cfg.get("channels", [])) if notif_cfg.get("channels") else
        "No channels configured."
    )
    if st.button("Send actionable signals now") and actionable:
        notifier = Notifier(
            channels=notif_cfg.get("channels", []),
            email=EmailConfig(**(notif_cfg.get("email") or {})),
            telegram=TelegramConfig(**(notif_cfg.get("telegram") or {})),
        )
        notifier.send_signals(actionable, header="Trading signals")
        st.success(f"Dispatched {len(actionable)} signals.")

st.markdown(
    "<small style='color:#8b949e;'>⚠️ Educational use only. No automated orders. "
    "Stop = Entry − ATR×Multiplier. Position sized by fixed-fractional risk.</small>",
    unsafe_allow_html=True,
)
