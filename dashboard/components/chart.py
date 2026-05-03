"""Plotly candlestick + indicator chart used on the detail view."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_chart(df: pd.DataFrame, ticker: str, signal=None) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} – Candlestick", "Volume", "RSI (14)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        ),
        row=1, col=1,
    )

    for col, color, name in [
        ("ema_50", "#d29922", "EMA 50"),
        ("ema_200", "#58a6ff", "EMA 200"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name,
                           line=dict(color=color, width=1.5), opacity=0.9),
                row=1, col=1,
            )

    if "bb_upper" in df.columns:
        for col, name, dash in [
            ("bb_upper", "BB Upper", "dot"),
            ("bb_mid", "BB Mid", "dash"),
            ("bb_lower", "BB Lower", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name,
                           line=dict(color="#8b949e", width=1, dash=dash), opacity=0.6),
                row=1, col=1,
            )

    if signal and signal.is_actionable:
        fig.add_hline(y=signal.entry, line_color="#58a6ff", line_dash="dot",
                      annotation_text=f"Entry {signal.entry:.2f}", row=1, col=1)
        fig.add_hline(y=signal.stop_loss, line_color="#f85149", line_dash="dot",
                      annotation_text=f"Stop {signal.stop_loss:.2f}", row=1, col=1)
        fig.add_hline(y=signal.take_profit, line_color="#3fb950", line_dash="dot",
                      annotation_text=f"TP {signal.take_profit:.2f}", row=1, col=1)

    if "Volume" in df.columns:
        colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color=colors, opacity=0.7),
            row=2, col=1,
        )

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
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#e6edf3"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=680,
    )
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)
    return fig
