"""The validation package exposes its public API at the top level."""

from __future__ import annotations

import tradinglib.validation as v


def test_public_api_present() -> None:
    for name in (
        "walk_forward",
        "WalkForwardResult",
        "SignalFn",
        "grid_search",
        "SearchResult",
        "expand_grid",
        "anchored_windows",
        "rolling_windows",
        "parameter_sensitivity",
        "metrics_by_regime",
        "vol_regime",
    ):
        assert hasattr(v, name), name
