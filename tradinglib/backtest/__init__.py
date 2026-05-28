"""Unified backtest engine — two interchangeable front-ends sharing one
PnL/metrics core.

- :func:`run_backtest` (vectorized) — pass a price series and a target-
  position series; fast and simple.
- :func:`run_event_backtest` (event-driven) — pass a sequence of
  :class:`Bar` events and a :class:`Strategy` callback; supports stateful
  per-bar logic. Returns the same :class:`BacktestResult` shape.
"""

from tradinglib.backtest.engine import BacktestResult, run_backtest
from tradinglib.backtest.event_engine import (
    Bar,
    EventEngine,
    Strategy,
    bars_from_dataframe,
    run_event_backtest,
)
from tradinglib.backtest.metrics import compute_metrics

__all__ = [
    "BacktestResult",
    "Bar",
    "EventEngine",
    "Strategy",
    "bars_from_dataframe",
    "compute_metrics",
    "run_backtest",
    "run_event_backtest",
]
