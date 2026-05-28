"""Unified backtest engine. Vectorized for bar-level strategies; event-driven
mode (forthcoming) for microstructure models."""

from tradinglib.backtest.engine import BacktestResult, run_backtest
from tradinglib.backtest.metrics import compute_metrics

__all__ = ["BacktestResult", "compute_metrics", "run_backtest"]
