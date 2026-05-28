"""Family-aware data preview panel.

Each model family has different input series. The vectorized models
expose ``close`` plus signal-relevant overlays (e.g. moving averages,
predicted returns), while the alt-data and microstructure models pair
``close`` with a separate panel for the non-price series (search
interest, smoothed OFI).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render(model_meta: dict[str, Any], out: dict[str, Any]) -> None:
    """Render the data preview panel for one model run."""
    family = model_meta.get("family", "")
    data = out["data"]
    symbol = out.get("symbol", "")

    st.subheader("Input data")

    if family == "classical":
        _render_price_with_mas(data, symbol)
    elif family == "ml":
        _render_price_with_prediction(data, symbol)
    elif family == "alt-data":
        _render_price_with_companion(
            data,
            companion_col="interest",
            companion_label="Search interest",
            symbol=symbol,
        )
    elif family == "microstructure":
        _render_price_with_companion(
            data,
            companion_col="ofi_smoothed",
            companion_label="OFI (smoothed)",
            symbol=symbol,
            bands=(0.2, -0.2),
        )
    else:
        _render_default(data, symbol)


def _render_price_with_mas(data: pd.DataFrame, symbol: str) -> None:
    """SMA-style: close + fast/slow moving averages on a single panel."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close", line={"color": "black"}))
    if "fast_ma" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["fast_ma"],
                name="Fast SMA",
                line={"color": "#2563eb", "width": 1},
            )
        )
    if "slow_ma" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["slow_ma"],
                name="Slow SMA",
                line={"color": "#ea580c", "width": 1},
            )
        )
    fig.update_layout(
        height=380,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        yaxis_title=f"{symbol} close",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_price_with_prediction(data: pd.DataFrame, symbol: str) -> None:
    """ML-style: close in the upper panel, predicted return in the lower."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["close"], name="Close", line={"color": "black"}),
        row=1,
        col=1,
    )
    if "predicted_return" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["predicted_return"],
                name="Predicted next-day return",
                line={"color": "#7c3aed", "width": 1},
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line={"color": "black", "width": 0.5, "dash": "dot"}, row=2, col=1)
    fig.update_layout(
        height=440,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        showlegend=True,
    )
    fig.update_yaxes(title_text=f"{symbol} close", row=1, col=1)
    fig.update_yaxes(title_text="Predicted return (log)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_price_with_companion(
    data: pd.DataFrame,
    companion_col: str,
    companion_label: str,
    symbol: str,
    bands: tuple[float, float] | None = None,
) -> None:
    """Two-panel layout: price on top, companion series on bottom.

    Used for alt-data (search interest) and microstructure (OFI) where
    the second series isn't directly comparable to price.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["close"], name="Close", line={"color": "black"}),
        row=1,
        col=1,
    )
    if companion_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[companion_col],
                name=companion_label,
                line={"color": "#2563eb", "width": 1.2},
            ),
            row=2,
            col=1,
        )
        if bands is not None:
            fig.add_hline(
                y=bands[0], line={"color": "green", "width": 1, "dash": "dash"}, row=2, col=1
            )
            fig.add_hline(
                y=bands[1], line={"color": "red", "width": 1, "dash": "dash"}, row=2, col=1
            )
            fig.add_hline(y=0.0, line={"color": "black", "width": 0.5}, row=2, col=1)
    fig.update_layout(
        height=460,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        showlegend=True,
    )
    fig.update_yaxes(title_text=f"{symbol} close", row=1, col=1)
    fig.update_yaxes(title_text=companion_label, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_default(data: pd.DataFrame, symbol: str) -> None:
    """Fallback when the family is unknown — plot every column."""
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col))
    fig.update_layout(
        height=380,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        yaxis_title=symbol,
    )
    st.plotly_chart(fig, use_container_width=True)
