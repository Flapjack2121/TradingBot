"""Trade-detail card with Confirm / Skip buttons."""
from __future__ import annotations

import streamlit as st


def render_trade_card(signal, repo, key_prefix: str = "") -> None:
    side_class = "buy" if signal.is_actionable else "wait"
    reasons_html = "".join(
        f"<li style='color:{'#3fb950' if v else '#f85149'};'>"
        f"{'✓' if v else '✗'} {k}</li>"
        for k, v in signal.reasons.items()
    )
    st.markdown(
        f"""
        <div class="signal-card {side_class}">
            <div class="strategy">{signal.strategy}</div>
            <div class="ticker">{signal.ticker}
                <span style="float:right;font-size:0.85rem;color:#8b949e;">
                    confidence {signal.confidence:.0%}
                </span>
            </div>
            <div class="price">
                Entry <b>{signal.entry}</b>  ·
                SL <b style="color:#f85149;">{signal.stop_loss}</b>  ·
                TP <b style="color:#3fb950;">{signal.take_profit}</b>
            </div>
            <div style="font-size:0.85rem;color:#8b949e;">
                Units {signal.units} · Risk €{signal.risk_amount} · Pos €{signal.position_value}
            </div>
            <ul style="font-size:0.85rem;list-style:none;padding-left:0;margin-top:8px;">
                {reasons_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not signal.is_actionable:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Confirm Trade", key=f"{key_prefix}-confirm-{signal.ticker}-{signal.strategy}"):
            t = repo.add_trade(
                ticker=signal.ticker,
                strategy=signal.strategy,
                side=signal.side,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                units=signal.units,
                risk_amount=signal.risk_amount,
                position_value=signal.position_value,
                r_multiple=signal.r_multiple,
                extras=signal.extras,
            )
            st.success(f"Trade #{t.id} recorded as PENDING.")
    with col2:
        if st.button("📈 Mark as OPEN", key=f"{key_prefix}-open-{signal.ticker}-{signal.strategy}"):
            from database import TradeStatus
            t = repo.add_trade(
                ticker=signal.ticker,
                strategy=signal.strategy,
                side=signal.side,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                units=signal.units,
                risk_amount=signal.risk_amount,
                position_value=signal.position_value,
                r_multiple=signal.r_multiple,
                status=TradeStatus.OPEN,
                extras=signal.extras,
            )
            st.success(f"Trade #{t.id} recorded as OPEN.")
    with col3:
        st.caption("Manual confirmation only — no auto-execution.")
