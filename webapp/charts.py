# webapp/charts.py
"""Registry of named Plotly figure-builders over a BacktestRun.

Adding a panel = writing a ``@register("name")`` function. The run view and the
HTMX partial enumerate ``BUILDERS``; no central switchboard to edit. Ported from
the Streamlit ``app/ui/*`` views, decoupled from Streamlit.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go

from tradinglib.eval.trades import trades_from_position
from tradinglib.service.run import BacktestRun

Builder = Callable[[BacktestRun], go.Figure]
BUILDERS: dict[str, Builder] = {}

# Palette — chosen to read on BOTH the bone (light) and night (dark) themes,
# since charts are rendered server-side once per run and can't restyle on a
# client-side theme flip. Transparent backgrounds let the page theme show
# through; the warm mid-grey axis/grid stays legible either way.
UP = "#2f9e54"
DOWN = "#cf3b32"
ACCENT = "#c8852f"
NEUTRAL = "#8a8478"
_FONT = "JetBrains Mono, ui-monospace, SFMono-Regular, monospace"
_GRID = "rgba(138,132,120,0.22)"


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        BUILDERS[name] = fn
        return fn

    return deco


def build(name: str, run: BacktestRun) -> go.Figure:
    if name not in BUILDERS:
        raise KeyError(name)
    return BUILDERS[name](run)


def build_all(run: BacktestRun) -> dict[str, go.Figure]:
    return {name: fn(run) for name, fn in BUILDERS.items()}


def _layout(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(
        margin={"t": 30, "b": 30, "l": 0, "r": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": _FONT, "color": NEUTRAL, "size": 11},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        **kw,
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    return fig


@register("equity")
def _equity(run: BacktestRun) -> go.Figure:
    res = run.result
    prices = run.data["close"]
    initial = res.config["initial_capital"]
    buy_hold = (1.0 + prices.pct_change().fillna(0.0)).cumprod() * initial
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=res.equity_curve.index,
            y=res.equity_curve.values,
            name="Strategy",
            line={"color": UP, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy_hold.index,
            y=buy_hold.values,
            name="Buy & hold",
            line={"color": NEUTRAL, "width": 1.5, "dash": "dash"},
        )
    )
    return _layout(fig, height=380, yaxis_title="Equity ($)")


@register("drawdown")
def _drawdown(run: BacktestRun) -> go.Figure:
    equity = run.result.equity_curve
    drawdown = equity / equity.cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            line={"color": DOWN},
            fillcolor="rgba(207,59,50,0.15)",
            name="Drawdown",
        )
    )
    return _layout(fig, height=280, yaxis_title="Drawdown", yaxis_tickformat=".0%")


@register("rolling_sharpe")
def _rolling_sharpe(run: BacktestRun, window: int = 63) -> go.Figure:
    r = run.result.returns
    ppy = run.result.config.get("periods_per_year", 252)
    roll = r.rolling(window)
    sharpe = (roll.mean() / roll.std(ddof=0)) * (ppy**0.5)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sharpe.index,
            y=sharpe.values,
            name=f"Rolling Sharpe ({window})",
            line={"color": ACCENT},
        )
    )
    fig.add_hline(y=0.0, line={"color": NEUTRAL, "width": 0.5, "dash": "dot"})
    return _layout(fig, height=280, yaxis_title="Sharpe")


@register("returns_hist")
def _returns_hist(run: BacktestRun) -> go.Figure:
    r = run.result.returns
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r.values, nbinsx=60, marker={"color": ACCENT}, name="Returns"))
    return _layout(fig, height=280, xaxis_title="Per-bar return", yaxis_title="Count", bargap=0.02)


@register("monthly_heatmap")
def _monthly_heatmap(run: BacktestRun) -> go.Figure:
    r = run.result.returns
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    df = pd.DataFrame({"ret": monthly.values}, index=monthly.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    grid = df.pivot_table(index="year", columns="month", values="ret")
    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[str(m) for m in grid.columns],
            y=[str(y) for y in grid.index],
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar={"tickformat": ".0%"},
        )
    )
    return _layout(fig, height=300, xaxis_title="Month", yaxis_title="Year")


@register("exposure")
def _exposure(run: BacktestRun) -> go.Figure:
    pos = run.result.position
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pos.index,
            y=pos.values,
            fill="tozeroy",
            name="Position",
            line={"color": "#3a6ea5"},
            fillcolor="rgba(58,110,165,0.15)",
        )
    )
    return _layout(fig, height=240, yaxis_title="Position (fraction)")


@register("trades_table")
def _trades_table(run: BacktestRun) -> go.Figure:
    trades = trades_from_position(run.result.position, run.data["close"])
    if trades.empty:
        fig = go.Figure()
        fig.add_annotation(text="No completed trades in this window", showarrow=False)
        return _layout(fig, height=200)
    header = ["Entry", "Exit", "Side", "Entry px", "Exit px", "PnL", "Bars"]
    cells = [
        [t.strftime("%Y-%m-%d") for t in trades["entry_time"]],
        [t.strftime("%Y-%m-%d") for t in trades["exit_time"]],
        list(trades["side"]),
        [f"{v:.2f}" for v in trades["entry_price"]],
        [f"{v:.2f}" for v in trades["exit_price"]],
        [f"{v:.2f}" for v in trades["pnl"]],
        list(trades["duration"]),
    ]
    fig = go.Figure(
        go.Table(
            header={
                "values": header,
                "fill_color": "rgba(138,132,120,0.18)",
                "font": {"color": NEUTRAL, "family": _FONT},
                "line_color": _GRID,
                "align": "left",
            },
            cells={
                "values": cells,
                "fill_color": "rgba(0,0,0,0)",
                "font": {"color": NEUTRAL, "family": _FONT},
                "line_color": _GRID,
                "align": "left",
            },
        )
    )
    return _layout(fig, height=320)
