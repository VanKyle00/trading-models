"""Backtest performance metrics — kept identical across every model so results
are comparable. See ``docs/methodology.md`` for assumptions and conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """Standard performance metrics on a per-bar return series.

    Returns a plain dict (JSON-serializable) so each model can dump it to
    ``results/metrics.json`` without further conversion.
    """
    if len(returns) == 0:
        return _empty_metrics()

    mean = float(returns.mean())
    std = float(returns.std(ddof=0))

    sharpe = float(np.sqrt(periods_per_year) * mean / std) if std > 0 else 0.0

    downside_std = float(returns.clip(upper=0.0).std(ddof=0))
    sortino = float(np.sqrt(periods_per_year) * mean / downside_std) if downside_std > 0 else 0.0

    # Annualized return from compounding the realized returns
    total_growth = float((1.0 + returns).prod())
    ann_return = (
        float(total_growth ** (periods_per_year / len(returns)) - 1.0) if total_growth > 0 else -1.0
    )

    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    max_drawdown = float(drawdowns.min())

    # Hit rate measured only over bars where the strategy was actually active
    nonzero = returns[returns != 0.0]
    hit_rate = float((nonzero > 0).mean()) if len(nonzero) > 0 else 0.0

    return {
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_bars": len(returns),
    }


def _empty_metrics() -> dict:
    return {
        "annualized_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "n_bars": 0,
    }
