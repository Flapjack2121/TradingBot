"""Trade-journal stats panel."""
from __future__ import annotations

import streamlit as st


def render_stats(stats: dict) -> None:
    if not stats or stats.get("trades", 0) == 0:
        st.info("No trades logged yet — confirm a signal to start your journal.")
        return
    cols = st.columns(6)
    cols[0].metric("Trades", stats.get("trades", 0))
    cols[1].metric("Open", stats.get("open", 0))
    cols[2].metric("Closed", stats.get("closed", 0))
    cols[3].metric("Win rate", f"{stats.get('win_rate', 0):.1f}%")
    cols[4].metric("Total PnL €", f"{stats.get('total_pnl', 0):,.2f}")
    cols[5].metric("Avg R", stats.get("avg_R") if stats.get("avg_R") is not None else "—")
