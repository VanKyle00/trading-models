# webapp/charts.py
"""Registry of named Plotly figure-builders over a BacktestRun.

Adding a panel = writing a ``@register("name")`` function. The run view and the
HTMX partial enumerate ``BUILDERS``; no central switchboard to edit. Ported from
the Streamlit ``app/ui/*`` views, decoupled from Streamlit.
"""

from __future__ import annotations

from collections.abc import Callable

import plotly.graph_objects as go

from tradinglib.service.run import BacktestRun

Builder = Callable[[BacktestRun], go.Figure]
BUILDERS: dict[str, Builder] = {}


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
    fig.update_layout(margin={"t": 30, "b": 30, "l": 0, "r": 0}, **kw)
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
            line={"color": "#dc2626"},
            name="Drawdown",
        )
    )
    return _layout(fig, height=280, yaxis_title="Drawdown", yaxis_tickformat=".0%")
