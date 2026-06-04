"""Options-specific panels: payoff curve and Monte Carlo P&L distribution.

Rendered only when a model's run_for_gui output carries the optional
``payoff`` / ``simulation`` keys, so non-options models are unaffected.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render(out: dict[str, Any]) -> None:
    if "payoff" in out:
        _render_payoff(out["payoff"])
    if "simulation" in out:
        _render_simulation(out["simulation"])


def _render_payoff(payoff: dict[str, Any]) -> None:
    st.subheader("Option payoff (single leg)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=payoff["spots"], y=payoff["values"], name="Value today"))
    fig.add_trace(
        go.Scatter(
            x=payoff["spots"],
            y=payoff["expiry_values"],
            name="Value at expiry",
            line={"dash": "dash"},
        )
    )
    fig.add_vline(x=payoff["strike"], line_dash="dot", line_color="gray")
    fig.update_layout(
        height=320,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        xaxis_title="Underlying spot",
        yaxis_title="Position value ($)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_simulation(sim: Any) -> None:
    st.subheader("Monte Carlo P&L distribution")
    if getattr(sim, "truncated", False):
        st.caption("⚠️ path count was capped to bound memory; showing the capped sample.")

    # SimulationResult always populates percentiles {5, 25, 50, 75, 95}.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prob. of profit", f"{sim.prob_of_profit:.0%}")
    c2.metric("Median P&L", f"${sim.percentiles[50]:,.0f}")
    c3.metric("Mean P&L", f"${sim.mean:,.0f}")
    c4.metric("Expected shortfall (5%)", f"${sim.expected_shortfall:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim.pnl_distribution, nbinsx=60, name="P&L"))
    for level in (5, 50, 95):
        fig.add_vline(
            x=sim.percentiles[level],
            line_dash="dot",
            annotation_text=f"p{level}",
            line_color="gray",
        )
    fig.update_layout(
        height=340,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        xaxis_title="Terminal P&L ($)",
        yaxis_title="Paths",
    )
    st.plotly_chart(fig, use_container_width=True)
