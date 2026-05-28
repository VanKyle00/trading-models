"""Family-aware data preview panel.

Each model family has different input series. The vectorized models
expose ``close`` plus signal-relevant overlays (e.g. moving averages,
predicted returns), while the alt-data and microstructure models pair
``close`` with a separate panel for the non-price series (search
interest, smoothed OFI).

Every chart also overlays **buy / sell markers** derived from the
backtest's ``position`` series — green up-triangles where the strategy
opened or added to a position, red down-triangles where it cut or
reversed. These are computed identically for every model so they're
directly comparable.
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
    position = out["result"].position

    st.subheader("Input data — with strategy entry / exit points")

    if family == "classical":
        _render_price_with_mas(data, symbol, position)
    elif family == "ml":
        _render_price_with_prediction(data, symbol, position)
    elif family == "alt-data":
        _render_price_with_companion(
            data,
            companion_col="interest",
            companion_label="Search interest",
            symbol=symbol,
            position=position,
        )
    elif family == "microstructure":
        _render_price_with_companion(
            data,
            companion_col="ofi_smoothed",
            companion_label="OFI (smoothed)",
            symbol=symbol,
            position=position,
            bands=(0.2, -0.2),
        )
    else:
        _render_default(data, symbol, position)

    _render_trade_legend(position)


def _trade_markers(position: pd.Series, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (buys, sells) — price values at bars where position changed.

    A bar where the position size *increased* relative to the prior bar
    is a buy (added long or covered short); a *decreased* position is a
    sell (cut long or opened short).
    """
    prev = position.shift(1).fillna(0.0)
    diff = position - prev
    aligned_prices = prices.reindex(position.index)
    buys = aligned_prices.where(diff > 0).dropna()
    sells = aligned_prices.where(diff < 0).dropna()
    return buys, sells


def _add_trade_markers(
    fig: go.Figure,
    position: pd.Series,
    prices: pd.Series,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Overlay buy/sell scatter markers on a Plotly figure."""
    buys, sells = _trade_markers(position, prices)
    target_kwargs: dict[str, int] = {}
    if row is not None and col is not None:
        target_kwargs = {"row": row, "col": col}

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys.values,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 9,
                    "color": "#22c55e",
                    "line": {"width": 1, "color": "#15803d"},
                },
                name="Buy",
                hovertemplate="Buy @ %{y:.2f}<br>%{x}<extra></extra>",
            ),
            **target_kwargs,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells.values,
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "size": 9,
                    "color": "#ef4444",
                    "line": {"width": 1, "color": "#b91c1c"},
                },
                name="Sell",
                hovertemplate="Sell @ %{y:.2f}<br>%{x}<extra></extra>",
            ),
            **target_kwargs,
        )


def _render_trade_legend(position: pd.Series) -> None:
    """Print a tiny line counting the trades in the window."""
    prev = position.shift(1).fillna(0.0)
    diff = position - prev
    n_buys = int((diff > 0).sum())
    n_sells = int((diff < 0).sum())
    if n_buys == 0 and n_sells == 0:
        st.caption("No position changes in this window — strategy stayed flat.")
    else:
        st.caption(
            f"🟢 **{n_buys}** entry-up events · 🔴 **{n_sells}** entry-down events "
            "(green = position increased, red = position decreased)"
        )


# ---------- family-specific renderers -------------------------------------


def _render_price_with_mas(data: pd.DataFrame, symbol: str, position: pd.Series) -> None:
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
    _add_trade_markers(fig, position, data["close"])
    fig.update_layout(
        height=420,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        yaxis_title=f"{symbol} close",
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_price_with_prediction(data: pd.DataFrame, symbol: str, position: pd.Series) -> None:
    """ML-style: close in the upper panel, predicted return in the lower."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["close"], name="Close", line={"color": "black"}),
        row=1,
        col=1,
    )
    _add_trade_markers(fig, position, data["close"], row=1, col=1)
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
        height=480,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        showlegend=True,
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    fig.update_yaxes(title_text=f"{symbol} close", row=1, col=1)
    fig.update_yaxes(title_text="Predicted return (log)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_price_with_companion(
    data: pd.DataFrame,
    companion_col: str,
    companion_label: str,
    symbol: str,
    position: pd.Series,
    bands: tuple[float, float] | None = None,
) -> None:
    """Two-panel layout: price on top with buy/sell markers, companion below."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["close"], name="Close", line={"color": "black"}),
        row=1,
        col=1,
    )
    _add_trade_markers(fig, position, data["close"], row=1, col=1)
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
        height=500,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        showlegend=True,
        legend={"orientation": "h", "y": 1.05, "x": 0},
    )
    fig.update_yaxes(title_text=f"{symbol} close", row=1, col=1)
    fig.update_yaxes(title_text=companion_label, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_default(data: pd.DataFrame, symbol: str, position: pd.Series) -> None:
    """Fallback when the family is unknown — plot every column."""
    fig = go.Figure()
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col))
    if "close" in data.columns:
        _add_trade_markers(fig, position, data["close"])
    fig.update_layout(
        height=420,
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        yaxis_title=symbol,
    )
    st.plotly_chart(fig, use_container_width=True)
