"""Validation & overfitting harness — walk-forward, search, sensitivity."""

from tradinglib.validation.search import SearchResult, expand_grid, grid_search
from tradinglib.validation.sensitivity import (
    metrics_by_regime,
    parameter_sensitivity,
    vol_regime,
)
from tradinglib.validation.splits import anchored_windows, rolling_windows
from tradinglib.validation.stats import benjamini_hochberg_fdr, bootstrap_t_test
from tradinglib.validation.walk_forward import SignalFn, WalkForwardResult, walk_forward

__all__ = [
    "SearchResult",
    "SignalFn",
    "WalkForwardResult",
    "anchored_windows",
    "benjamini_hochberg_fdr",
    "bootstrap_t_test",
    "expand_grid",
    "grid_search",
    "metrics_by_regime",
    "parameter_sensitivity",
    "rolling_windows",
    "vol_regime",
    "walk_forward",
]
