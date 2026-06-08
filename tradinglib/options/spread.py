"""Synthetic bid/ask spread model for option fills.

The engine fills option legs by crossing this spread: buys at the ask, sells at
the bid. The half-spread is a fraction of premium that widens for
out-of-the-money and short-dated options, with an absolute per-share floor
(``min_tick``) because even cheap options cost a minimum to cross.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpreadModel(Protocol):
    min_tick: float

    def half_spread_frac(self, mid: float, m: float, dte: int) -> float: ...


@dataclass(frozen=True)
class NoSpread(SpreadModel):
    """Frictionless fills (mid). Backward-compat + before/after comparison."""

    min_tick: float = 0.0

    def half_spread_frac(self, mid: float, m: float, dte: int) -> float:
        return 0.0


@dataclass(frozen=True)
class ParametricSpread(SpreadModel):
    """Half-spread fraction widening for OTM / short-DTE, with a per-share floor."""

    base: float = 0.01
    otm_penalty: float = 0.05
    short_dte_penalty: float = 0.02
    max_frac: float = 0.5
    min_tick: float = 0.05

    def half_spread_frac(self, mid: float, m: float, dte: int) -> float:
        d = max(dte, 1)
        frac = self.base + self.otm_penalty * abs(m) + self.short_dte_penalty / math.sqrt(d)
        return min(max(frac, 0.0), self.max_frac)
