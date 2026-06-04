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
            line={"color": "#7c3aed"},
        )
    )
    fig.add_hline(y=0.0, line={"color": "black", "width": 0.5, "dash": "dot"})
    return _layout(fig, height=280, yaxis_title="Sharpe")


@register("returns_hist")
def _returns_hist(run: BacktestRun) -> go.Figure:
    r = run.result.returns
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r.values, nbinsx=60, marker={"color": "#2563eb"}, name="Returns"))
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
            line={"color": "#0891b2"},
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
    fig = go.Figure(go.Table(header={"values": header}, cells={"values": cells}))
    return _layout(fig, height=320)
