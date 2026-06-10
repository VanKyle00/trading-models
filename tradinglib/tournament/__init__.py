"""Per-ticker walk-forward strategy tournament (sub-project B of the
FA-tournament-tickets design). Only strategies that survive out-of-sample
with a globally-deflated Sharpe may produce trade levels."""

from tradinglib.tournament.levels import Levels
from tradinglib.tournament.run import (
    StrategyVerdict,
    TournamentConfig,
    TournamentResult,
    run_tournament,
)
from tradinglib.tournament.strategies import STRATEGIES, StrategyDef, register

__all__ = [
    "STRATEGIES",
    "Levels",
    "StrategyDef",
    "StrategyVerdict",
    "TournamentConfig",
    "TournamentResult",
    "register",
    "run_tournament",
]
