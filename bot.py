"""Trading Signal Dashboard — main Streamlit application.

Run with:

    streamlit run bot.py

Top-level tabs are one per *trading mode* (Swing, Day) plus a shared Journal.
Inside each mode tab the same set of sub-tabs is rendered (Overview, one
per strategy, Actionable, Chart, Backtest, Strategy Lab) with mode-specific
data and BUY parameters.
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
from strategies import (
    REGISTRY, SignalEngine, STRATEGY_LABELS, SIDE_BUY, SIDE_WAIT, SIDE_AVOID,
    registry_for_mode, registry_for_asset_class,
)
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
                  mode_key: str) -> dict:
    """Full scan: fetch data per asset class, run all strategies, tag signals.

    The mode (``"swing"`` / ``"day"``) drives the bar interval, history depth,
    indicator periods, and the named strategies & parameters. Cache is keyed
    by mode so swing and day scans don't trample each other.
    """
    dm = get_data_manager()
    cfg = settings.load_config()
    mode = cfg.get("modes", {}).get(mode_key, {})

    interval = mode.get("interval", "1d")
    period = mode.get("period", "2y")
    indicator_cfg = mode.get("indicators", cfg.get("indicators", {}))
    strat_params = mode.get("strategies", {})
    enabled = list(strat_params.get("enabled", list(registry_for_mode(mode_key).keys())))

    engine = build_engine(account, risk_pct, atr_mult, tp_r, enabled, strat_params)

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

    # Drop signals where the strategy is not documented for that asset class.
    # We keep AVOID-style "downtrend" verdicts even if the suitability chip
    # says no — they're informational. Only filter actionable BUY/WAIT.
    signals = [
        s for s in signals
        if REGISTRY.get(s.strategy) is None
        or REGISTRY[s.strategy].suits(s.asset_class)
    ]

    regime = dm.market_regime(sma_window=cfg.get("regime", {}).get("spy_sma", 200),
                               force_refresh=force)
    return {
        "frames": frames, "signals": signals, "regime": regime,
        "selection": selection, "ticker_class": ticker_class,
        "mode": {"key": mode_key, "interval": interval, "period": period,
                  "label": mode.get("label", mode_key.title()),
                  "description": mode.get("description", ""),
                  "max_hold_bars": mode.get("max_hold_bars", 60),
                  "strategies": strat_params,
                  "enabled": enabled,
                  "indicators": indicator_cfg},
    }


# ── Sidebar (shared across modes) ────────────────────────────────────────
cfg = settings.load_config()
risk_cfg = cfg.get("risk", {})
strategy_cfg_global = cfg.get("strategies", {})
modes_cfg = cfg.get("modes", {})
mode_keys = list(modes_cfg.keys()) or ["swing"]

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
    st.caption(
        "🧠 **Strategies** are mode-specific and configured in `config.yaml` "
        "(`modes.<mode>.strategies.enabled`)."
    )
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
st.info(
    "📚 All strategies in this dashboard are documented systems from published "
    "trader/academic literature with historical edge — Minervini, Connors, "
    "Dennis (Turtle), Bollinger, Crabel, and the institutional VWAP / floor-"
    "trader Pivot tradition. **Past performance does not guarantee future "
    "returns.** Use the **Strategy Lab** tab to validate on your basket and "
    "horizon before committing capital.",
    icon="ℹ️",
)
st.markdown("---")

if not selected_universes:
    st.warning("Pick at least one asset universe in the sidebar.")
    st.stop()


# ── Helpers ──────────────────────────────────────────────────────────────
def _summary_metrics(df: pd.DataFrame, container) -> None:
    cols = container.columns(5)
    cols[0].metric("Tickers", df["Ticker"].nunique() if not df.empty else 0)
    cols[1].metric("BUY",   int((df["Signal"] == SIDE_BUY).sum())   if not df.empty else 0)
    cols[2].metric("WAIT",  int((df["Signal"] == SIDE_WAIT).sum())  if not df.empty else 0)
    cols[3].metric("AVOID", int((df["Signal"] == SIDE_AVOID).sum()) if not df.empty else 0)
    cols[4].metric("Asset classes",
                    df["Asset Class"].nunique() if "Asset Class" in df.columns and not df.empty else 0)


def _filtered(df: pd.DataFrame, sides: list[str] | None = None,
               asset_filter: str = "All") -> pd.DataFrame:
    out = df.copy()
    if sides:
        out = out[out["Signal"].isin(sides)]
    if asset_filter != "All" and "Asset Class" in out.columns:
        out = out[out["Asset Class"] == asset_filter]
    return out


def render_strategy_subtab(strategy_name: str, df_all: pd.DataFrame,
                            signals_list: list, asset_classes: list[str],
                            mode_key: str) -> None:
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
            key=f"sidefilter_{mode_key}_{strategy_name}",
        )
    with c2:
        asset_filter = st.selectbox(
            "Asset class", ["All"] + asset_classes,
            key=f"acfilter_{mode_key}_{strategy_name}",
        )
    with c3:
        sort_by = st.selectbox(
            "Sort by", ["Confidence ↓", "Signal", "Ticker", "Asset Class"],
            key=f"sort_{mode_key}_{strategy_name}",
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

    if not view.empty:
        st.markdown("##### 🔎 Per-ticker rationale")
        pick = st.selectbox(
            "Select ticker for full rationale",
            view["Ticker"].tolist(),
            key=f"pick_{mode_key}_{strategy_name}",
        )
        sig = next((s for s in signals_list
                    if s.ticker == pick and s.strategy == strategy_name), None)
        if sig is not None:
            color = {SIDE_BUY: "#3fb950", SIDE_WAIT: "#8b949e",
                     SIDE_AVOID: "#f85149"}[sig.side]
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


def render_mode_view(mode_key: str) -> None:
    """Render every sub-tab for one trading mode."""
    with st.spinner(f"Scanning markets in {modes_cfg.get(mode_key, {}).get('label', mode_key)} mode…"):
        result = scan_pipeline(
            tuple(selected_universes), max_us, force,
            account_size, risk_pct, atr_mult, tp_r, mode_key,
        )

    regime = result["regime"]
    frames: dict[str, pd.DataFrame] = result["frames"]
    signals = result["signals"]
    ticker_class = result["ticker_class"]
    selection = result["selection"]
    mode_info = result["mode"]

    # Mode + regime banner
    mode_color = "#bc8cff" if mode_info["key"] == "day" else "#58a6ff"
    regime_label = regime.get("regime", "unknown")
    spy_close = regime.get("spy_close")
    spy_sma = regime.get("spy_sma")
    regime_emoji = {"bull": "🐂", "bear": "🐻"}.get(regime_label, "❓")
    regime_color = {"bull": "#3fb950", "bear": "#f85149"}.get(regime_label, "#8b949e")

    st.markdown(
        f"<div style='border-left:4px solid {mode_color};padding:8px 14px;"
        f"background:#161b22;border-radius:6px;margin-bottom:12px;'>"
        f"<b style='color:{mode_color};'>{mode_info['label']}</b> · "
        f"interval <code>{mode_info['interval']}</code> · "
        f"history <code>{mode_info['period']}</code> · "
        f"max hold <code>{mode_info['max_hold_bars']} bars</code><br>"
        f"<small style='color:#8b949e;'>{mode_info['description']}</small><br>"
        f"<span style='color:{regime_color};font-weight:bold;'>"
        f"{regime_emoji} Market regime: {regime_label.upper()}</span>"
        + (f"  &nbsp;<small style='color:#8b949e;'>SPY {spy_close:.2f} vs SMA200 {spy_sma:.2f}</small>"
           if spy_close and spy_sma else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Build per-mode engine for risk fill / actionable filtering
    enabled = mode_info["enabled"]
    engine = build_engine(account_size, risk_pct, atr_mult, tp_r,
                           enabled, mode_info["strategies"])
    df_all = engine.to_dataframe(signals)
    actionable = engine.actionable(signals)
    asset_classes = sorted(set(ticker_class.values())) or ["—"]

    # Per-strategy "info card": label, source, suitability chips, params.
    with st.expander("🧠 Active strategies — sources, suitability & parameters"):
        st.caption(
            "ℹ️ All systems below are documented in published trader/academic "
            "literature with historical edge. **No guarantee of future returns.** "
            "Use the Strategy Lab tab to verify on the data you care about."
        )
        for sname in enabled:
            cls = REGISTRY.get(sname)
            if cls is None:
                continue
            params = mode_info["strategies"].get(sname, {}) or {}
            classes_for_chip = cls.asset_classes
            chip_html = " ".join(
                f"<span style='background:#1a3a2a;color:#3fb950;"
                f"padding:2px 8px;border-radius:10px;font-size:0.75rem;"
                f"margin-right:4px;'>{ac}</span>"
                for ac in classes_for_chip
            )
            st.markdown(
                f"**{cls.label}** {chip_html}",
                unsafe_allow_html=True,
            )
            if cls.description:
                st.caption(cls.description)
            if cls.source:
                st.caption(f"📚 *{cls.source}*")
            if params:
                st.json(params)
            else:
                st.caption("(defaults)")
            st.markdown("---")

    # Sub-tabs ------------------------------------------------------------
    sub_names = (
        ["📊 Overview"]
        + [STRATEGY_LABELS.get(n, n) for n in enabled]
        + ["🎯 Actionable", "🕯️ Chart", "🧪 Backtest", "📚 Strategy Lab"]
    )
    sub_tabs = st.tabs(sub_names)

    # Overview ------------------------------------------------------------
    with sub_tabs[0]:
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

    # Per-strategy sub-tabs ----------------------------------------------
    for i, name in enumerate(enabled, start=1):
        with sub_tabs[i]:
            st.markdown(f"### {STRATEGY_LABELS.get(name, name)}")
            render_strategy_subtab(name, df_all, signals, asset_classes, mode_key)

    sub_offset = 1 + len(enabled)

    # Actionable ----------------------------------------------------------
    with sub_tabs[sub_offset]:
        st.markdown(f"### Actionable trades ({len(actionable)})")
        if not actionable:
            st.info("No actionable BUY signals on this scan.")
        repo = get_repo()
        asset_pick = st.selectbox("Asset class", ["All"] + asset_classes,
                                    key=f"act_ac_{mode_key}")
        candidates = sorted(actionable, key=lambda x: -x.confidence)
        if asset_pick != "All":
            candidates = [s for s in candidates if s.asset_class == asset_pick]
        for s in candidates:
            render_trade_card(s, repo, key_prefix=f"act_{mode_key}")

    # Chart ---------------------------------------------------------------
    with sub_tabs[sub_offset + 1]:
        st.markdown("### Interactive chart")
        selectable = sorted(frames.keys())
        if not selectable:
            st.info("No data fetched.")
        else:
            sel = st.selectbox("Ticker", selectable, key=f"chart_ticker_{mode_key}")
            df_sel = frames[sel]
            sigs_sel = [s for s in signals if s.ticker == sel]
            c1, c2 = st.columns([2, 1])
            with c2:
                for s in sigs_sel:
                    color = {SIDE_BUY: "#3fb950", SIDE_WAIT: "#8b949e",
                             SIDE_AVOID: "#f85149"}[s.side]
                    st.markdown(
                        f"<div style='border:1px solid {color};border-radius:8px;"
                        f"padding:8px 12px;margin-bottom:6px;'>"
                        f"<b style='color:{color};'>"
                        f"{STRATEGY_LABELS.get(s.strategy, s.strategy)}</b> · "
                        f"{s.side} ({s.confidence:.0%})<br>"
                        f"<small style='color:#8b949e;'>{s.rationale}</small></div>",
                        unsafe_allow_html=True,
                    )
            with c1:
                best = next((s for s in sigs_sel if s.is_actionable),
                            sigs_sel[0] if sigs_sel else None)
                st.plotly_chart(build_chart(df_sel, sel, signal=best),
                                  use_container_width=True)

    # Backtest ------------------------------------------------------------
    with sub_tabs[sub_offset + 2]:
        from backtest import Backtester
        bt_cfg = cfg.get("backtest", {})
        st.markdown(f"### Single-symbol backtest · {mode_info['label']}")
        st.caption(
            f"Uses the active mode's bars ({mode_info['interval']}, "
            f"{mode_info['period']}) and strategy parameters."
        )
        if not frames:
            st.info("No data loaded.")
        else:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                bt_ticker = st.selectbox("Ticker", sorted(frames.keys()),
                                           key=f"bt_ticker_{mode_key}")
            with col_b:
                bt_strat_name = st.selectbox(
                    "Strategy", enabled, key=f"bt_strat_{mode_key}",
                    format_func=lambda n: STRATEGY_LABELS.get(n, n),
                )
            with col_c:
                bt_cash = st.number_input(
                    "Initial cash €", min_value=100,
                    value=int(bt_cfg.get("initial_cash", 10000)),
                    key=f"bt_cash_{mode_key}",
                )

            if st.button("Run backtest", key=f"runbt_{mode_key}"):
                with st.spinner("Backtesting…"):
                    strat_cls = REGISTRY[bt_strat_name]
                    strat = strat_cls(mode_info["strategies"].get(bt_strat_name, {}))
                    bt = Backtester(
                        initial_cash=bt_cash,
                        risk_per_trade=risk_pct,
                        atr_stop_multiplier=atr_mult,
                        take_profit_r_multiple=tp_r,
                        commission_pct=bt_cfg.get("commission_pct", 0.0005),
                        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
                        max_hold_bars=mode_info["max_hold_bars"],
                    )
                    bt_res = bt.run(bt_ticker, frames[bt_ticker], strat)
                st.json(bt_res.stats)
                if not bt_res.equity_curve.empty:
                    st.line_chart(bt_res.equity_curve, height=320)
                if not bt_res.trades.empty:
                    st.markdown("##### Trades")
                    st.dataframe(bt_res.trades, use_container_width=True,
                                  hide_index=True)

    # Strategy Lab --------------------------------------------------------
    with sub_tabs[sub_offset + 3]:
        from backtest import StrategyLab
        bt_cfg = cfg.get("backtest", {})

        st.markdown(f"### 📚 Strategy Lab · {mode_info['label']}")
        st.caption(
            "Run a strategy across a basket of tickers and see aggregated "
            "performance: win rate, CAGR, Sharpe, profit factor, drawdown, "
            "and per-ticker breakdown."
        )
        if mode_info["key"] == "day":
            st.warning(
                "⚠️ Day mode active. yfinance only serves up to ~730 days of "
                "hourly bars, so the lab caps the period accordingly. For "
                "10+ year backtests use the Swing tab."
            )

        mode_strat_keys = list(registry_for_mode(mode_key).keys()) or enabled
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            lab_strat_name = st.selectbox(
                "Strategy", mode_strat_keys,
                key=f"lab_strat_{mode_key}",
                format_func=lambda n: STRATEGY_LABELS.get(n, n),
            )
        with col2:
            if mode_info["key"] == "day":
                lab_period = st.selectbox(
                    "History", ["60d", "180d", "365d", "730d"],
                    index=2, key=f"lab_period_{mode_key}",
                )
            else:
                lab_period = st.selectbox(
                    "History", ["5y", "10y", "15y", "20y", "max"],
                    index=1, key=f"lab_period_{mode_key}",
                )
        with col3:
            lab_universe_key = st.selectbox(
                "Basket", list(U.UNIVERSES.keys()),
                format_func=lambda k: U.label_for(k),
                key=f"lab_basket_{mode_key}",
            )
        with col4:
            lab_n = st.slider("Tickers in basket", 5, 50, 15, 5,
                                key=f"lab_n_{mode_key}")

        col5, col6, col7 = st.columns(3)
        with col5:
            lab_cash = st.number_input(
                "Cash per ticker (€)", min_value=500,
                value=int(bt_cfg.get("initial_cash", 10000)), step=500,
                key=f"lab_cash_{mode_key}",
            )
        with col6:
            lab_max_hold = st.slider(
                "Max hold (bars)", 5, 250, mode_info["max_hold_bars"], 5,
                key=f"lab_hold_{mode_key}",
            )
        with col7:
            st.metric("Estimated total capital",
                       f"€{lab_cash * lab_n:,.0f}")

        st.caption(
            "ℹ️ First run downloads long-history data — for 15 tickers expect "
            "1–2 minutes. Subsequent runs use the cache and complete in seconds."
        )

        if st.button("🚀 Run Strategy Lab", key=f"run_lab_{mode_key}"):
            basket_tickers = U.get_tickers(lab_universe_key)[:lab_n]
            if not basket_tickers:
                st.error("Empty basket — pick another asset class.")
            else:
                strat_cls = REGISTRY[lab_strat_name]
                strat = strat_cls(mode_info["strategies"].get(lab_strat_name, {}))
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
                    lab_result = lab.run(
                        strat, basket_tickers,
                        period=lab_period,
                        interval=mode_info["interval"],
                        indicator_cfg=mode_info["indicators"],
                        progress=_progress,
                    )
                progress_bar.empty()

                stats = lab_result.portfolio_stats
                if not stats:
                    st.error("No backtest results — check tickers and period.")
                else:
                    st.success(
                        f"Backtested **{stats.get('tickers_run', 0)}** tickers over "
                        f"**{stats.get('years', 0)}** years with "
                        f"**{stats.get('trades', 0)}** trades."
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
                    m[2].metric("Calmar",
                                  f"{stats.get('calmar', 0):.2f}" if stats.get('calmar') else "—")
                    m[3].metric("Recovery",
                                  f"{stats.get('recovery_factor', 0):.2f}" if stats.get('recovery_factor') else "—")

                    st.markdown("#### 📉 Drawdown")
                    m = st.columns(2)
                    m[0].metric("Max Drawdown",         f"{stats.get('max_drawdown_pct', 0):.2f}%")
                    m[1].metric("Max DD duration (bars)", stats.get('max_dd_duration_bars', 0))

                    st.markdown("#### 🎯 Trade quality")
                    m = st.columns(4)
                    m[0].metric("Win Rate",      f"{stats.get('win_rate_pct', 0):.1f}%")
                    m[1].metric("Profit Factor",
                                  f"{stats.get('profit_factor', 0):.2f}" if stats.get('profit_factor') else "—")
                    m[2].metric("Payoff Ratio",
                                  f"{stats.get('payoff_ratio', 0):.2f}" if stats.get('payoff_ratio') else "—")
                    m[3].metric("Expectancy/trade",
                                  f"€{stats.get('expectancy_eur', 0):,.2f}")

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
                    if not lab_result.portfolio_equity.empty:
                        eq = lab_result.portfolio_equity
                        chart_df = pd.DataFrame({
                            "Portfolio": eq,
                            "Drawdown %": (eq / eq.cummax() - 1) * 100,
                        })
                        st.line_chart(chart_df["Portfolio"], height=320)
                        st.area_chart(chart_df["Drawdown %"], height=180,
                                       color="#f85149")

                    st.markdown("#### 🔍 Per-ticker breakdown")
                    ticker_table = lab_result.per_ticker_table
                    if not ticker_table.empty:
                        cols_to_show = [
                            "ticker", "trades", "win_rate_pct", "total_return_pct",
                            "cagr_pct", "max_drawdown_pct", "sharpe", "sortino",
                            "profit_factor", "avg_R", "expectancy_eur",
                        ]
                        cols_to_show = [c for c in cols_to_show
                                          if c in ticker_table.columns]
                        st.dataframe(ticker_table[cols_to_show],
                                      use_container_width=True, hide_index=True)

                    st.markdown("#### 🧾 Pooled trades (chronological)")
                    if not lab_result.pooled_trades.empty:
                        st.dataframe(lab_result.pooled_trades.tail(200),
                                      use_container_width=True, hide_index=True)
                        if st.button("📤 Export all trades to CSV",
                                       key=f"export_lab_{mode_key}"):
                            path = Reporter().save_trades(lab_result.pooled_trades, fmt="csv")
                            st.success(f"Saved {path}")

    return signals  # so caller can aggregate notifications later


def render_journal_tab() -> None:
    repo = get_repo()
    st.markdown("### 📒 Trade journal")
    st.caption(
        "All trades you confirmed across both modes live here — Swing and "
        "Day positions share one history."
    )
    render_stats(repo.stats())
    df_trades = repo.to_dataframe()
    if df_trades.empty:
        st.info("No trades recorded yet. Confirm one in any mode's Actionable tab.")
        return

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
    col_e1, _ = st.columns(2)
    if col_e1.button("📤 Export trades to CSV"):
        path = Reporter().save_trades(df_trades, fmt="csv")
        st.success(f"Saved {path}")


# ── Top-level tabs (one per mode + shared journal) ───────────────────────
top_tab_labels = ([modes_cfg.get(mk, {}).get("label", mk.title()) for mk in mode_keys]
                   + ["📒 Journal"])
top_tabs = st.tabs(top_tab_labels)

mode_signals: dict[str, list] = {}
for i, mk in enumerate(mode_keys):
    with top_tabs[i]:
        sigs = render_mode_view(mk)
        if sigs is not None:
            mode_signals[mk] = sigs

with top_tabs[-1]:
    render_journal_tab()


# ── Notifications (manual trigger) ───────────────────────────────────────
st.markdown("---")
with st.expander("📨 Notifications"):
    notif_cfg = cfg.get("notifications", {})
    if notif_cfg.get("channels"):
        st.caption("Configure in `config.yaml`. Channels enabled: "
                   + ", ".join(notif_cfg.get("channels", [])))
    else:
        st.caption("No channels configured. Add email/telegram in `config.yaml` to enable.")

    notif_mode = st.radio("Send signals from", mode_keys,
                            format_func=lambda k: modes_cfg.get(k, {}).get("label", k),
                            horizontal=True, key="notif_mode")
    sigs_for_notif = mode_signals.get(notif_mode, [])
    actionable_for_notif = [s for s in sigs_for_notif if s.is_actionable]
    st.caption(f"{len(actionable_for_notif)} actionable signal(s) ready to send.")
    if st.button("Send actionable signals now") and actionable_for_notif:
        notifier = Notifier(
            channels=notif_cfg.get("channels", []),
            email=EmailConfig(**(notif_cfg.get("email") or {})),
            telegram=TelegramConfig(**(notif_cfg.get("telegram") or {})),
        )
        header = f"Trading signals — {modes_cfg.get(notif_mode, {}).get('label', notif_mode)}"
        notifier.send_signals(actionable_for_notif, header=header)
        st.success(f"Dispatched {len(actionable_for_notif)} signals.")

st.markdown(
    "<small style='color:#8b949e;'>⚠️ Educational use only. No automated orders. "
    "Stop = Entry − ATR×Multiplier. Position sized by fixed-fractional risk.</small>",
    unsafe_allow_html=True,
)
