"""Vectorized backtest engine for bar-level strategies.

Given a price series and a target-position series, computes the equity curve,
per-bar net returns, and standard performance metrics under linear
transaction-cost assumptions.

Conventions
-----------
- Bars are equally spaced; the engine does not try to handle calendar gaps.
- A signal at bar ``t`` is lagged one bar before multiplying by returns, which
  prevents look-ahead bias. By default trades fill at the **next bar's open**
  (``fill="next_open"``, which requires ``open_prices``): the entry bar earns
  ``open[t] -> close[t]`` and the overnight gap is not captured. Pass
  ``fill="decision_close"`` to fill at the decision bar's own close (optimistic;
  close-to-close).
- A position of ``1.0`` means fully invested; ``-1.0`` fully short; ``0.0`` flat.
  Fractional values size partially.
- Transaction costs are linear in turnover, in basis points (``1 bp = 0.01%``).
  Slippage is symmetric.

An event-driven front-end (:mod:`tradinglib.backtest.event_engine`) generates
signals via a per-bar callback and feeds them through this same vectorized core,
so PnL and metrics are identical.
"""

from __future__ import annotations

import warnings
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
    open_prices: pd.Series | None = None,
    fill: str = "next_open",
    execution_prices: pd.Series | None = None,
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
    open_prices:
        Per-bar open prices (same index as ``prices``). Required when
        ``fill="next_open"``: a position change is filled at this bar's open, so
        the entry bar earns ``open -> close``. Held bars stay close-to-close.
    fill:
        ``"next_open"`` (default) or ``"decision_close"``. ``"decision_close"``
        fills at the decision bar's own close (the prior, optimistic behavior).
    execution_prices:
        Deprecated alias for ``open_prices`` (implies ``fill="next_open"``).
    """
    if execution_prices is not None:
        warnings.warn(
            "execution_prices= is deprecated; pass open_prices= instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if open_prices is not None:
            raise ValueError(
                "pass either open_prices= or the deprecated execution_prices=, not both"
            )
        open_prices = execution_prices
        fill = "next_open"

    if fill not in ("next_open", "decision_close"):
        raise ValueError(f"fill must be 'next_open' or 'decision_close', got {fill!r}")
    if fill == "next_open" and open_prices is None:
        raise ValueError(
            "fill='next_open' (the default) requires open_prices; pass the open "
            "series, or set fill='decision_close' for close-to-close fills"
        )

    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals must share the same index")
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to compute a return")

    if open_prices is not None and not prices.index.equals(open_prices.index):
        raise ValueError("prices and open_prices must share the same index")

    price_returns = prices.pct_change().fillna(0.0)

    # Lag the signal: a decision made looking at bar t's close pays off
    # starting at bar t+1. This is the single most common source of
    # look-ahead bugs in backtests — keep it explicit.
    position = signals.shift(1).fillna(0.0)

    prev_position = position.shift(1).fillna(0.0)
    turnover = (position - prev_position).abs()

    if fill == "next_open":
        # A position change in force at bar t was decided at close[t-1] and is
        # filled at bar t's OPEN. The entry bar earns open[t] -> close[t]; held
        # bars stay close-to-close. Removes the optimism of filling at the very
        # close used to make the decision.
        entered = turnover > 0.0
        entry_returns = (prices / open_prices - 1.0).fillna(0.0)
        price_returns = price_returns.where(~entered, entry_returns)

    gross_returns = position * price_returns

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
            "execution": fill,
        },
    )
