"""Vectorized backtest engine for bar-level strategies.

Given a price series and a target-position series, computes the equity curve,
per-bar net returns, and standard performance metrics under linear
transaction-cost assumptions.

Conventions
-----------
- Bars are equally spaced; the engine does not try to handle calendar gaps.
- A signal at bar ``t`` is executed at the close of bar ``t`` — PnL on that
  decision accrues over the bar ``t → t+1``. This is enforced by lagging the
  signal one bar before multiplying by returns, which prevents look-ahead bias.
- A position of ``1.0`` means fully invested in the asset; ``-1.0`` means
  fully short; ``0.0`` means flat. Fractional values for partial sizing.
- Transaction costs are linear in turnover, expressed in basis points
  (``1 bp = 0.01%``). Slippage is symmetric.

For tick-level microstructure models, an event-driven engine will live next
to this one (``event_engine.py``) — not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradinglib.backtest.metrics import compute_metrics


@dataclass
class BacktestResult:
    """Standardized backtest output — every model in this repo returns one."""

    equity_curve: pd.Series
    returns: pd.Series
    position: pd.Series
    turnover: pd.Series
    metrics: dict
    config: dict = field(default_factory=dict)


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
    n_trials: int = 1,
) -> BacktestResult:
    """Run a vectorized backtest of a single-asset strategy.

    Parameters
    ----------
    prices:
        Close prices indexed by time.
    signals:
        Target position at each bar; same index as ``prices``.
    initial_capital:
        Starting cash, used only to scale the equity curve.
    fee_bps:
        Per-unit-turnover commission in basis points.
    slippage_bps:
        Per-unit-turnover slippage in basis points.
    periods_per_year:
        Bars per year — used by :func:`compute_metrics` to annualize Sharpe /
        Sortino. 252 for daily, ~98 280 for 1-minute US-equity bars.
    n_trials:
        Number of independent strategy configurations tried to arrive at this
        one. Forwarded to :func:`compute_metrics` to deflate the Sharpe; ``1``
        (the default) leaves the Deflated Sharpe equal to the Probabilistic
        Sharpe.
    """
    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals must share the same index")
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to compute a return")

    price_returns = prices.pct_change().fillna(0.0)

    # Lag the signal: a decision made looking at bar t's close pays off
    # starting at bar t+1. This is the single most common source of
    # look-ahead bugs in backtests — keep it explicit.
    position = signals.shift(1).fillna(0.0)

    gross_returns = position * price_returns

    prev_position = position.shift(1).fillna(0.0)
    turnover = (position - prev_position).abs()
    cost_per_unit_turnover = (fee_bps + slippage_bps) / 10_000.0
    cost_drag = turnover * cost_per_unit_turnover

    net_returns = gross_returns - cost_drag
    equity_curve = (1.0 + net_returns).cumprod() * initial_capital

    metrics = compute_metrics(
        net_returns, equity_curve, periods_per_year=periods_per_year, n_trials=n_trials
    )

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        position=position,
        turnover=turnover,
        metrics=metrics,
        config={
            "initial_capital": initial_capital,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "periods_per_year": periods_per_year,
            "n_trials": n_trials,
        },
    )
