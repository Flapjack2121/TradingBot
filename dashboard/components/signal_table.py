"""Renders the main signal overview table."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def _style_signal(val: str) -> str:
    if val == "BUY":
        return "background-color:#1a3a2a;color:#3fb950;font-weight:bold;border-radius:4px;padding:2px 8px;"
    if val == "WAIT":
        return "background-color:#2a2a2a;color:#8b949e;border-radius:4px;padding:2px 8px;"
    return "color:#8b949e;"


def render_signal_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No signals to display — check the universe and refresh.")
        return

    show = df.copy()
    for col in ("Entry", "Stop Loss", "Take Profit", "ATR"):
        if col in show.columns:
            show[col] = show[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    for col in ("Risk €", "Position €"):
        if col in show.columns:
            show[col] = show[col].apply(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    if "As Of" in show.columns:
        show["As Of"] = pd.to_datetime(show["As Of"]).dt.strftime("%Y-%m-%d")

    styled = (
        show.style
        .map(_style_signal, subset=["Signal"])
        .set_properties(**{
            "background-color": "#161b22", "color": "#e6edf3",
            "border": "1px solid #30363d",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#21262d"),
                ("color", "#58a6ff"),
                ("border", "1px solid #30363d"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
