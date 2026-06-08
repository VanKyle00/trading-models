"""The options package exposes the new surface/spread API at the top level."""
from __future__ import annotations

import tradinglib.options as o


def test_surface_and_spread_exported() -> None:
    for name in (
        "VolSurface", "FlatSurface", "ParametricSurface", "SurfaceParams",
        "realistic_surface", "realized_vol",
        "SpreadModel", "NoSpread", "ParametricSpread",
    ):
        assert hasattr(o, name), name
