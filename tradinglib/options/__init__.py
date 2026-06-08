"""Options pricing, Greeks, instruments, and Monte Carlo simulation.

Pure, dependency-light primitives (numpy + scipy.stats) used by the options
backtest engine in ``tradinglib.backtest.options_engine`` and the seed model
under ``models/options/``.
"""

from __future__ import annotations

from tradinglib.options.spread import NoSpread, ParametricSpread, SpreadModel
from tradinglib.options.surface import (
    EventVolSurface,
    FlatSurface,
    ParametricSurface,
    SurfaceParams,
    VolSurface,
    realistic_surface,
    realized_vol,
)

__all__ = [
    "EventVolSurface",
    "FlatSurface",
    "NoSpread",
    "ParametricSpread",
    "ParametricSurface",
    "SpreadModel",
    "SurfaceParams",
    "VolSurface",
    "realistic_surface",
    "realized_vol",
]
