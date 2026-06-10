"""Tournament strategy registry — import surface preserved.

Family modules self-register on import; ``_core`` holds the registry and
shared helpers so family modules never import this package root (no cycle).
"""

from tradinglib.tournament.strategies import (  # noqa: F401  (registration side effects)
    breakout,
    mean_reversion,
    trend,
)
from tradinglib.tournament.strategies._core import (
    STRATEGIES,
    LevelsFn,
    StrategyDef,
    TournamentSignalFn,
    _full_history,
    _hold_between,
    register,
)

__all__ = [
    "STRATEGIES",
    "LevelsFn",
    "StrategyDef",
    "TournamentSignalFn",
    "_full_history",
    "_hold_between",
    "register",
]
