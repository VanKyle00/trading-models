"""Monte Carlo simulation for options strategies.

:func:`gbm_paths` generates geometric-Brownian-motion underlying paths under
the risk-neutral measure. :func:`run_simulation` runs a strategy across many
paths and aggregates to a :class:`SimulationResult` distribution.

Memory: paths are ``float32`` and the simulation aggregates to per-path P&L
without retaining per-leg histories, keeping peak memory well under the ~1 GB
Streamlit Community Cloud cap. Callers should still bound ``n_paths``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglib.backtest.options_engine import OptionsStrategy, run_options_backtest

TRADING_DAYS_PER_YEAR = 252


def gbm_paths(
    spot: float,
    vol: float,
    rate: float,
    days: int,
    n_paths: int,
    *,
    steps_per_day: int = 1,
    seed: int | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """Simulate GBM underlying paths. Returns shape ``(n_paths, days*steps_per_day + 1)``.

    Column 0 is the starting spot. ``dt`` is in years (``1 / (252*steps_per_day)``),
    so the terminal horizon is ``days / 252`` years.
    """
    rng = np.random.default_rng(seed)
    n_steps = days * steps_per_day
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * steps_per_day)
    drift = (rate - 0.5 * vol * vol) * dt
    diffusion = vol * math.sqrt(dt)

    z: np.ndarray = rng.standard_normal((n_paths, n_steps)).astype(dtype)
    log_increments = drift + diffusion * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths: np.ndarray = (spot * np.exp(log_paths)).astype(dtype)
    start: np.ndarray = np.full((n_paths, 1), spot, dtype=dtype)
    return np.concatenate([start, paths], axis=1)


@dataclass
class SimulationResult:
    """Distribution of strategy P&L across simulated paths."""

    pnl_distribution: np.ndarray          # terminal P&L per path (initial capital subtracted)
    percentiles: dict[int, float]         # {5, 25, 50, 75, 95} -> P&L
    prob_of_profit: float
    expected_shortfall: float             # mean P&L of the worst 5% of paths
    mean: float
    std: float
    sample_paths: np.ndarray              # a handful of underlying paths for plotting
    truncated: bool                       # True if n_paths was clamped to max_paths


def run_simulation(
    strategy_factory: Callable[[], OptionsStrategy],
    *,
    spot: float,
    vol: float,
    rate: float,
    days: int,
    n_paths: int,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    seed: int | None = None,
    max_paths: int = 20_000,
    n_sample_paths: int = 40,
) -> SimulationResult:
    """Run ``strategy_factory()`` (a fresh strategy per path) across GBM paths.

    Returns a :class:`SimulationResult`. ``n_paths`` is clamped to ``max_paths``
    to bound memory/runtime; the returned ``truncated`` flag records whether the
    clamp fired so callers can surface it.
    """
    truncated = n_paths > max_paths
    n_paths = min(n_paths, max_paths)

    paths = gbm_paths(spot, vol, rate, days, n_paths, seed=seed)
    index = pd.bdate_range("2024-01-01", periods=paths.shape[1])

    pnl: np.ndarray = np.empty(n_paths, dtype=np.float64)
    for i in range(n_paths):
        price_series = pd.Series(paths[i].astype(np.float64), index=index)
        result = run_options_backtest(
            price_series,
            strategy_factory(),
            vol=vol,
            rate=rate,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        pnl[i] = float(result.equity_curve.iloc[-1]) - initial_capital

    pct_levels = [5, 25, 50, 75, 95]
    pct_values = np.percentile(pnl, pct_levels)
    percentiles = {lvl: float(v) for lvl, v in zip(pct_levels, pct_values, strict=True)}
    worst_5pct = pnl[pnl <= percentiles[5]]
    expected_shortfall = float(worst_5pct.mean()) if worst_5pct.size else float(pnl.min())

    return SimulationResult(
        pnl_distribution=pnl,
        percentiles=percentiles,
        prob_of_profit=float((pnl > 0).mean()),
        expected_shortfall=expected_shortfall,
        mean=float(pnl.mean()),
        std=float(pnl.std(ddof=0)),
        sample_paths=paths[:n_sample_paths],
        truncated=truncated,
    )
