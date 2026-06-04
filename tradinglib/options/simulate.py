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

import numpy as np

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
