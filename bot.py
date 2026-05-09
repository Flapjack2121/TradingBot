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
from data import universes as U
from strategies import REGISTRY, SignalEngine, STRATEGY_LABELS, SIDE_BUY, SIDE_WAIT, SIDE_AVOID
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


def build_engine(account: float, risk_pct: float, atr_mult: float, tp_r: float,
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


@st.cache_data(ttl=900, show_spinner=False)
def scan_pipeline(universe_keys: tuple[str, ...], max_us: int, force: bool,
                  account: float, risk_pct: float, atr_mult: float, tp_r: float,
                  enabled: tuple[str, ...], mode_key: str) -> dict:
    """Full scan: fetch data per asset class, run all strategies, tag signals.

    The mode (``"swing"`` / ``"day"``) drives the bar interval, history depth,
    indicator periods, and per-strategy parameters. Cache is keyed by mode so
    swing and day scans don't trample each other.
    """
    dm = get_data_manager()
    cfg = settings.load_config()
    mode = cfg.get("modes", {}).get(mode_key, {})

    interval = mode.get("interval", "1d")
    period = mode.get("period", "2y")
    indicator_cfg = mode.get("indicators", cfg.get("indicators", {}))
    strat_params = mode.get("strategies", cfg.get("strategies", {}))

    engine = build_engine(account, risk_pct, atr_mult, tp_r, list(enabled), strat_params)

    selection = U.build_selection(list(universe_keys),
                                   max_per_universe={"us_stocks": max_us})
    ticker_class = U.ticker_to_asset_class(selection)
    all_tickers = sorted({t for ticks in selection.values() for t in ticks})

    frames = dm.bulk(
        all_tickers, force_refresh=force,
        period=period, interval=interval, indicator_cfg=indicator_cfg,
    )
    signals = engine.scan(frames)
    SignalEngine.apply_asset_classes(signals, ticker_class)

    regime = dm.market_regime(sma_window=cfg.get("regime", {}).get("spy_sma", 200),
                               force_refresh=force)
    return {
        "frames": frames, "signals": signals, "regime": regime,
        "selection": selection, "ticker_class": ticker_class,
        "mode": {"key": mode_key, "interval": interval, "period": period,
                  "label": mode.get("label", mode_key.title()),
                  "description": mode.get("description", "")},
    }


# ── Sidebar controls ─────────────────────────────────────────────────────
cfg = settings.load_config()
risk_cfg = cfg.get("risk", {})
strategy_cfg = cfg.get("strategies", {})

modes_cfg = cfg.get("modes", {})
mode_keys = list(modes_cfg.keys()) or ["swing"]
mode_label_for = lambda k: modes_cfg.get(k, {}).get("label", k.title())

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("### ⏱️ Trading Mode")
    mode_key = st.radio(
        "Timeframe", mode_keys,
        format_func=mode_label_for,
        index=0, key="trading_mode",
        horizontal=True,
    )
    mode_desc = modes_cfg.get(mode_key, {}).get("description", "")
    if mode_desc:
        st.caption(mode_desc)

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
    st.markdown("### 🌍 Asset Universes")
    universe_options = list(U.UNIVERSES.keys())
    universe_labels = {k: U.UNIVERSES[k]["label"] for k in universe_options}
    selected_universes = []
    default_selected = ["us_stocks", "forex", "commodities", "crypto"]
    for key in universe_options:
        if st.checkbox(universe_labels[key],
                        value=(key in default_selected),
                        key=f"uni_{key}"):
            selected_universes.append(key)
    max_us = st.slider("Max US stocks (S&P 500)", 10, 503, 50, 10)

    st.markdown("---")
    st.markdown("### 🧠 Strategies")
    enabled_default = strategy_cfg.get("enabled", list(REGISTRY.keys()))
    enabled = []
    for name in REGISTRY.keys():
        if st.checkbox(STRATEGY_LABELS.get(name, name),
                        value=name in enabled_default,
                        key=f"strat_{name}"):
            enabled.append(name)

    st.markdown("---")
    force = st.checkbox("Force refresh data", value=False)
    if st.button("🔄 Re-scan now"):
        st.cache_data.clear()
        st.rerun()


# ── Header ───────────────────────────────────────────────────────────────
st.markdown(f"# 📈 {cfg.get('dashboard', {}).get('title', 'Trading Signal Dashboard')}")
st.caption(
    f"Account: **€{account_size:,.0f}**  •  "
    f"Risk/Trade: **{risk_pct*100:.2f}%**  •  "
    f"Stop: **{atr_mult}× ATR**  •  TP: **{tp_r}R**  •  "
    "Human-in-the-loop — no automated execution."
)

if not selected_universes:
    st.warning("Pick at least one asset universe in the sidebar.")
    st.stop()
if not enabled:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

with st.spinner(f"Scanning markets in {mode_label_for(mode_key)} mode…"):
    result = scan_pipeline(
        tuple(selected_universes), max_us, force,
        account_size, risk_pct, atr_mult, tp_r, tuple(enabled), mode_key,
    )

regime = result["regime"]
frames: dict[str, pd.DataFrame] = result["frames"]
signals = result["signals"]
ticker_class = result["ticker_class"]
selection = result["selection"]
mode_info = result["mode"]

# From this point on, prefer the mode-specific strategy and indicator
# parameters over the legacy top-level config blocks. This way the Backtest
# and Strategy Lab tabs honour the same mode the user picked above.
mode_block = modes_cfg.get(mode_info["key"], {})
strategy_cfg = mode_block.get("strategies", strategy_cfg)
indicator_cfg_active = mode_block.get("indicators", cfg.get("indicators", {}))

mode_color = "#bc8cff" if mode_info["key"] == "day" else "#58a6ff"
st.markdown(
    f"<div style='border-left:4px solid {mode_color};padding:6px 14px;"
    f"background:#161b22;border-radius:6px;margin-bottom:12px;'>"
    f"<b style='color:{mode_color};'>{mode_info['label']}</b> · "
    f"interval <code>{mode_info['interval']}</code> · "
    f"history <code>{mode_info['period']}</code>"
    + (f" · <span style='color:#8b949e;'>{mode_info['description']}</span>"
       if mode_info.get('description') else "")
    + "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")


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


# ── Build full DataFrame once ────────────────────────────────────────────
engine = build_engine(account_size, risk_pct, atr_mult, tp_r, enabled, strategy_cfg)
df_all = engine.to_dataframe(signals)
actionable = engine.actionable(signals)
asset_classes = sorted(set(ticker_class.values())) or ["—"]


def _filtered(df: pd.DataFrame, sides: list[str] | None = None,
               asset_filter: str = "All") -> pd.DataFrame:
    out = df.copy()
    if sides:
        out = out[out["Signal"].isin(sides)]
    if asset_filter != "All" and "Asset Class" in out.columns:
        out = out[out["Asset Class"] == asset_filter]
    return out


def _summary_metrics(df: pd.DataFrame, container) -> None:
    cols = container.columns(5)
    cols[0].metric("Tickers", df["Ticker"].nunique() if not df.empty else 0)
    cols[1].metric("BUY",   int((df["Signal"] == SIDE_BUY).sum())   if not df.empty else 0)
    cols[2].metric("WAIT",  int((df["Signal"] == SIDE_WAIT).sum())  if not df.empty else 0)
    cols[3].metric("AVOID", int((df["Signal"] == SIDE_AVOID).sum()) if not df.empty else 0)
    cols[4].metric("Asset classes",
                    df["Asset Class"].nunique() if "Asset Class" in df.columns and not df.empty else 0)


def render_strategy_tab(strategy_name: str, df_all: pd.DataFrame) -> None:
    sub = df_all[df_all["Strategy"] == strategy_name].copy()
    if sub.empty:
        st.info("No signals for this strategy.")
        return

    _summary_metrics(sub, st)

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        side_filter = st.multiselect(
            "Signal", [SIDE_BUY, SIDE_WAIT, SIDE_AVOID],
            default=[SIDE_BUY, SIDE_WAIT, SIDE_AVOID],
            key=f"sidefilter_{strategy_name}",
        )
    with c2:
        asset_filter = st.selectbox(
            "Asset class", ["All"] + asset_classes,
            key=f"acfilter_{strategy_name}",
        )
    with c3:
        sort_by = st.selectbox(
            "Sort by", ["Confidence ↓", "Signal", "Ticker", "Asset Class"],
            key=f"sort_{strategy_name}",
        )

    view = _filtered(sub, side_filter, asset_filter)
    if sort_by == "Confidence ↓":
        view = view.sort_values("Confidence", ascending=False)
    elif sort_by == "Signal":
        view = view.sort_values(["Signal", "Confidence"], ascending=[True, False])
    elif sort_by == "Ticker":
        view = view.sort_values("Ticker")
    else:
        view = view.sort_values(["Asset Class", "Confidence"], ascending=[True, False])

    render_signal_table(view)

    # Per-ticker rationale picker
    if not view.empty:
        st.markdown("##### 🔎 Per-ticker rationale")
        pick = st.selectbox(
            "Select ticker for full rationale",
            view["Ticker"].tolist(),
            key=f"pick_{strategy_name}",
        )
        sig = next((s for s in signals
                    if s.ticker == pick and s.strategy == strategy_name), None)
        if sig is not None:
            color = {SIDE_BUY: "#3fb950", SIDE_WAIT: "#8b949e", SIDE_AVOID: "#f85149"}[sig.side]
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:8px 14px;"
                f"background:#161b22;border-radius:6px;'>"
                f"<b style='color:{color};'>{sig.side}</b> · "
                f"<span style='color:#8b949e;'>confidence {sig.confidence:.0%}"
                + (f" · {sig.asset_class}" if sig.asset_class else "")
                + f"</span><br>{sig.rationale}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Conditions checked"):
                for k, v in sig.reasons.items():
                    icon = "✅" if v else "❌"
                    st.markdown(f"- {icon} {k}")


# ── Tabs ─────────────────────────────────────────────────────────────────
tab_names = (
    ["📊 Overview"]
    + [STRATEGY_LABELS.get(n, n) for n in enabled]
    + ["🎯 Actionable", "🕯️ Chart", "📒 Journal", "🧪 Backtest", "📚 Strategy Lab"]
)
tabs = st.tabs(tab_names)

# Overview ----------------------------------------------------------------
with tabs[0]:
    st.markdown("### Market Overview")
    _summary_metrics(df_all, st)

    if not df_all.empty:
        st.markdown("##### Signal mix by strategy")
        mix = df_all.pivot_table(
            index="Strategy", columns="Signal", values="Ticker",
            aggfunc="count", fill_value=0,
        )
        for col in (SIDE_BUY, SIDE_WAIT, SIDE_AVOID):
            if col not in mix.columns:
                mix[col] = 0
        mix = mix[[SIDE_BUY, SIDE_WAIT, SIDE_AVOID]]
        st.dataframe(mix, use_container_width=True)

        st.markdown("##### Signal mix by asset class")
        if "Asset Class" in df_all.columns:
            mix_ac = df_all.pivot_table(
                index="Asset Class", columns="Signal", values="Ticker",
                aggfunc="count", fill_value=0,
            )
            for col in (SIDE_BUY, SIDE_WAIT, SIDE_AVOID):
                if col not in mix_ac.columns:
                    mix_ac[col] = 0
            mix_ac = mix_ac[[SIDE_BUY, SIDE_WAIT, SIDE_AVOID]]
            st.dataframe(mix_ac, use_container_width=True)

        st.markdown("##### Universe coverage")
        st.write({U.label_for(k): len(v) for k, v in selection.items()})

# Per-strategy tabs --------------------------------------------------------
for i, name in enumerate(enabled, start=1):
    with tabs[i]:
        st.markdown(f"### {STRATEGY_LABELS.get(name, name)}")
        render_strategy_tab(name, df_all)

# Actionable --------------------------------------------------------------
offset = 1 + len(enabled)
with tabs[offset]:
    st.markdown(f"### Actionable trades ({len(actionable)})")
    if not actionable:
        st.info("No actionable BUY signals on this scan.")
    repo = get_repo()
    asset_pick = st.selectbox("Asset class", ["All"] + asset_classes, key="act_ac")
    candidates = sorted(actionable, key=lambda x: -x.confidence)
    if asset_pick != "All":
        candidates = [s for s in candidates if s.asset_class == asset_pick]
    for s in candidates:
        render_trade_card(s, repo, key_prefix="act")

# Chart -------------------------------------------------------------------
with tabs[offset + 1]:
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
                color = {SIDE_BUY: "#3fb950", SIDE_WAIT: "#8b949e", SIDE_AVOID: "#f85149"}[s.side]
                st.markdown(
                    f"<div style='border:1px solid {color};border-radius:8px;"
                    f"padding:8px 12px;margin-bottom:6px;'>"
                    f"<b style='color:{color};'>{STRATEGY_LABELS.get(s.strategy, s.strategy)}</b> · "
                    f"{s.side} ({s.confidence:.0%})<br>"
                    f"<small style='color:#8b949e;'>{s.rationale}</small></div>",
                    unsafe_allow_html=True,
                )
        with c1:
            best = next((s for s in sigs_sel if s.is_actionable),
                        sigs_sel[0] if sigs_sel else None)
            st.plotly_chart(build_chart(df_sel, sel, signal=best), use_container_width=True)

# Journal -----------------------------------------------------------------
with tabs[offset + 2]:
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

# Backtest ----------------------------------------------------------------
with tabs[offset + 3]:
    st.markdown(f"### Single-symbol backtest · {mode_info['label']}")
    st.caption(
        f"Uses the active mode's bars ({mode_info['interval']}, {mode_info['period']}) "
        "and strategy parameters."
    )
    if not frames:
        st.info("No data loaded.")
    else:
        from backtest import Backtester
        bt_cfg = cfg.get("backtest", {})
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            bt_ticker = st.selectbox("Ticker", sorted(frames.keys()), key="bt_ticker")
        with col_b:
            bt_strat_name = st.selectbox("Strategy", enabled, key="bt_strat",
                                           format_func=lambda n: STRATEGY_LABELS.get(n, n))
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
                    max_hold_bars=mode_block.get("max_hold_bars", 60),
                )
                bt_res = bt.run(bt_ticker, frames[bt_ticker], strat)
            st.json(bt_res.stats)
            if not bt_res.equity_curve.empty:
                st.line_chart(bt_res.equity_curve, height=320)
            if not bt_res.trades.empty:
                st.markdown("##### Trades")
                st.dataframe(bt_res.trades, use_container_width=True, hide_index=True)

# Strategy Lab ------------------------------------------------------------
with tabs[offset + 4]:
    from backtest import StrategyLab

    st.markdown("### 📚 Strategy Lab — multi-year, multi-ticker validation")
    st.caption(
        "Run a strategy across a basket of tickers and see aggregated "
        "performance: win rate, CAGR, Sharpe, profit factor, drawdown, and "
        "per-ticker breakdown."
    )
    if mode_info["key"] == "day":
        st.warning(
            "⚠️ Day mode active. yfinance only serves up to ~730 days of "
            "hourly bars, so the lab caps the period accordingly. For "
            "10+ year backtests switch to Swing mode."
        )

    bt_cfg = cfg.get("backtest", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lab_strat_name = st.selectbox(
            "Strategy", list(REGISTRY.keys()), key="lab_strat",
            format_func=lambda n: STRATEGY_LABELS.get(n, n),
        )
    with col2:
        if mode_info["key"] == "day":
            # yfinance hard-caps hourly history at 730 days.
            lab_period = st.selectbox(
                "History", ["60d", "180d", "365d", "730d"],
                index=2, key="lab_period",
            )
        else:
            lab_period = st.selectbox(
                "History", ["5y", "10y", "15y", "20y", "max"],
                index=1, key="lab_period",
            )
    with col3:
        lab_universe_key = st.selectbox(
            "Basket", list(U.UNIVERSES.keys()),
            format_func=lambda k: U.label_for(k), key="lab_basket",
        )
    with col4:
        lab_n = st.slider("Tickers in basket", 5, 50, 15, 5, key="lab_n")

    col5, col6, col7 = st.columns(3)
    with col5:
        lab_cash = st.number_input(
            "Cash per ticker (€)", min_value=500,
            value=int(bt_cfg.get("initial_cash", 10000)), step=500, key="lab_cash",
        )
    with col6:
        lab_max_hold = st.slider("Max hold (bars)", 5, 250, 60, 5, key="lab_hold")
    with col7:
        st.metric("Estimated total capital",
                   f"€{lab_cash * lab_n:,.0f}")

    st.caption(
        "ℹ️ First run downloads long-history data from yfinance — for 15 tickers × 10y "
        "expect 1–2 minutes. Subsequent runs use the cache and complete in seconds."
    )

    if st.button("🚀 Run Strategy Lab", key="run_lab"):
        basket_tickers = U.get_tickers(lab_universe_key)[:lab_n]
        if not basket_tickers:
            st.error("Empty basket — pick another asset class.")
        else:
            strat_cls = REGISTRY[lab_strat_name]
            strat = strat_cls(strategy_cfg.get(lab_strat_name, {}))
            lab = StrategyLab(
                data_manager=get_data_manager(),
                initial_cash_per_ticker=lab_cash,
                risk_per_trade=risk_pct,
                atr_stop_multiplier=atr_mult,
                take_profit_r_multiple=tp_r,
                commission_pct=bt_cfg.get("commission_pct", 0.0005),
                slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
                max_hold_bars=lab_max_hold,
            )

            progress_bar = st.progress(0.0, text="Starting…")

            def _progress(i: int, n: int, ticker: str) -> None:
                pct = i / n if n else 1.0
                progress_bar.progress(min(pct, 1.0),
                                       text=f"{i}/{n} · {ticker}")

            with st.spinner("Running portfolio backtest…"):
                result = lab.run(
                    strat, basket_tickers,
                    period=lab_period,
                    interval=mode_info["interval"],
                    indicator_cfg=indicator_cfg_active,
                    progress=_progress,
                )
            progress_bar.empty()

            stats = result.portfolio_stats
            if not stats:
                st.error("No backtest results — check tickers and period.")
            else:
                st.success(
                    f"Backtested **{stats.get('tickers_run', 0)}** tickers over "
                    f"**{stats.get('years', 0)}** years with **{stats.get('trades', 0)}** trades."
                )

                st.markdown("#### 💰 Portfolio performance")
                m = st.columns(4)
                m[0].metric("Total Return", f"{stats.get('total_return_pct', 0):.2f}%")
                m[1].metric("CAGR",         f"{stats.get('cagr_pct', 0):.2f}%")
                m[2].metric("Volatility",   f"{stats.get('volatility_pct', 0):.2f}%")
                m[3].metric("Final Equity", f"€{stats.get('final_equity_total', 0):,.0f}")

                st.markdown("#### 🧮 Risk-adjusted")
                m = st.columns(4)
                m[0].metric("Sharpe",   f"{stats.get('sharpe', 0):.2f}")
                m[1].metric("Sortino",  f"{stats.get('sortino', 0):.2f}")
                m[2].metric("Calmar",   f"{stats.get('calmar', 0):.2f}" if stats.get('calmar') else "—")
                m[3].metric("Recovery", f"{stats.get('recovery_factor', 0):.2f}" if stats.get('recovery_factor') else "—")

                st.markdown("#### 📉 Drawdown")
                m = st.columns(2)
                m[0].metric("Max Drawdown",         f"{stats.get('max_drawdown_pct', 0):.2f}%")
                m[1].metric("Max DD duration (days)", stats.get('max_dd_duration_bars', 0))

                st.markdown("#### 🎯 Trade quality")
                m = st.columns(4)
                m[0].metric("Win Rate",      f"{stats.get('win_rate_pct', 0):.1f}%")
                m[1].metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}" if stats.get('profit_factor') else "—")
                m[2].metric("Payoff Ratio",  f"{stats.get('payoff_ratio', 0):.2f}" if stats.get('payoff_ratio') else "—")
                m[3].metric("Expectancy/trade", f"€{stats.get('expectancy_eur', 0):,.2f}")

                m = st.columns(4)
                m[0].metric("Avg Win",   f"€{stats.get('avg_win_eur', 0):,.2f}")
                m[1].metric("Avg Loss",  f"€{stats.get('avg_loss_eur', 0):,.2f}")
                m[2].metric("Largest Win",  f"€{stats.get('largest_win_eur', 0):,.2f}")
                m[3].metric("Largest Loss", f"€{stats.get('largest_loss_eur', 0):,.2f}")

                m = st.columns(3)
                m[0].metric("Total Trades", stats.get("trades", 0))
                m[1].metric("Avg R / trade", stats.get("avg_R", "—"))
                m[2].metric("Avg hold (days)", stats.get("avg_hold_days", "—"))

                st.markdown("#### 📈 Portfolio equity curve")
                if not result.portfolio_equity.empty:
                    eq = result.portfolio_equity
                    chart_df = pd.DataFrame({
                        "Portfolio": eq,
                        "Drawdown %": (eq / eq.cummax() - 1) * 100,
                    })
                    st.line_chart(chart_df["Portfolio"], height=320)
                    st.area_chart(chart_df["Drawdown %"], height=180,
                                   color="#f85149")

                st.markdown("#### 🔍 Per-ticker breakdown")
                ticker_table = result.per_ticker_table
                if not ticker_table.empty:
                    cols_to_show = [
                        "ticker", "trades", "win_rate_pct", "total_return_pct",
                        "cagr_pct", "max_drawdown_pct", "sharpe", "sortino",
                        "profit_factor", "avg_R", "expectancy_eur",
                    ]
                    cols_to_show = [c for c in cols_to_show if c in ticker_table.columns]
                    st.dataframe(ticker_table[cols_to_show], use_container_width=True,
                                  hide_index=True)

                st.markdown("#### 🧾 Pooled trades (all tickers, chronological)")
                if not result.pooled_trades.empty:
                    st.dataframe(result.pooled_trades.tail(200),
                                  use_container_width=True, hide_index=True)
                    if st.button("📤 Export all trades to CSV", key="export_lab"):
                        path = Reporter().save_trades(result.pooled_trades, fmt="csv")
                        st.success(f"Saved {path}")

# ── Notifications (manual trigger) ───────────────────────────────────────
st.markdown("---")
with st.expander("📨 Notifications"):
    notif_cfg = cfg.get("notifications", {})
    if notif_cfg.get("channels"):
        st.caption("Configure in `config.yaml`. Channels enabled: "
                   + ", ".join(notif_cfg.get("channels", [])))
    else:
        st.caption("No channels configured. Add email/telegram in `config.yaml` to enable.")
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
