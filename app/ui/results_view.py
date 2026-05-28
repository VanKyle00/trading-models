"""Backtest results panel — equity curve, drawdown, metrics table."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(out: dict[str, Any]) -> None:
    """Render the equity curve, metrics, and drawdown for one model run."""
    result = out["result"]
    data = out["data"]

    st.subheader("Backtest results")
    _render_equity_overlay(result, data["close"])

    mc1, mc2 = st.columns([1, 2])
    with mc1:
        st.markdown("**Metrics**")
        st.dataframe(_metrics_table(result.metrics), hide_index=True, use_container_width=True)
    with mc2:
        st.markdown("**Drawdown**")
        _render_drawdown(result.equity_curve)


def _render_equity_overlay(result: Any, prices: pd.Series) -> None:
    """Equity curve with a buy-and-hold benchmark on the same axis."""
    initial = result.config["initial_capital"]
    buy_hold = (1.0 + prices.pct_change().fillna(0.0)).cumprod() * initial

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name="Strategy",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy_hold.index,
            y=buy_hold.values,
            name="Buy & hold",
            line={"color": "gray", "width": 1.5, "dash": "dash"},
        )
    )
    fig.update_layout(
        height=380,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        yaxis_title="Equity ($)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for k, v in metrics.items():
        display = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows.append({"metric": k, "value": display})
    return pd.DataFrame(rows)


def _render_drawdown(equity: pd.Series) -> None:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            line={"color": "#dc2626"},
            name="Drawdown",
        )
    )
    fig.update_layout(
        height=300,
        margin={"t": 10, "b": 30, "l": 0, "r": 0},
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)
