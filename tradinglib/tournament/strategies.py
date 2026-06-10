"""Registry of tournament strategies — popular retail rules, long/short aware.

Each ``StrategyDef`` bundles a parameter grid, a walk-forward signal builder
and a ``levels`` builder turning the latest bars into tomorrow's concrete
entry/stop/target. ``stance`` flips the rule itself (a Donchian short sells
the low-break), not just the sign of the output. Adding a strategy = one
``register(StrategyDef(...))`` call — the tournament and the /models page
enumerate ``STRATEGIES``; nothing else to edit.

Signals are target-position series: ``{0, +1}`` for long stance, ``{0, -1}``
for short — exactly what ``run_backtest`` consumes. Builders warm indicators
with train history, deduped so in-sample scoring (train == test) works.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglib.features.technical import sma
from tradinglib.tournament.levels import Levels, direction, protective_stop, two_r_target

# (train, test, params, stance) -> target-position series indexed like test
TournamentSignalFn = Callable[[pd.DataFrame, pd.DataFrame, dict, str], pd.Series]
LevelsFn = Callable[[pd.DataFrame, dict, str], Levels]


@dataclass(frozen=True)
class StrategyDef:
    key: str
    name: str
    style: str  # "trend" | "breakout" | "mean_reversion"
    description: str  # plain-English rule; powers the /models page
    param_grid: dict[str, list]
    make_signal: TournamentSignalFn
    levels: LevelsFn


STRATEGIES: dict[str, StrategyDef] = {}


def register(sdef: StrategyDef) -> StrategyDef:
    if sdef.key in STRATEGIES:
        raise ValueError(f"duplicate strategy key {sdef.key!r}")
    STRATEGIES[sdef.key] = sdef
    return sdef


def _full_history(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train + test with the overlap deduped (in-sample scoring passes train as test)."""
    full = pd.concat([train, test])
    return full[~full.index.duplicated(keep="last")]


def _hold_between(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Flip-flop position state: 1 from an entry bar until the next exit bar.

    Returns ``{0, 1}`` regardless of stance; short-stance callers negate the
    result themselves (an IEEE ``-0.0`` on flat bars compares equal to ``0.0``
    everywhere downstream, so the sign of zero is irrelevant).
    """
    state = pd.Series(np.nan, index=entry.index)
    state[exit_] = 0.0
    state[entry] = 1.0  # entry wins when both fire on the same bar
    return state.ffill().fillna(0.0)


# --- sma_cross ---------------------------------------------------------------


def _sma_cross_signal(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.Series:
    closes = _full_history(train, test)["close"]
    fast = sma(closes, params["fast"])
    slow = sma(closes, params["slow"])
    pos = (fast > slow).astype(float) if direction(stance) > 0 else -(fast < slow).astype(float)
    return pos.loc[test.index]


def _sma_cross_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels:
    entry = float(bars["close"].iloc[-1])
    stop = protective_stop(bars, entry, stance)
    side = "above" if direction(stance) > 0 else "below"
    return Levels(
        entry=entry,
        entry_type="market",
        stop=stop,
        target=two_r_target(entry, stop),
        condition=(
            f"enter while SMA({params['fast']}) holds {side} SMA({params['slow']}); "
            "rule exits on the opposite cross"
        ),
    )


register(
    StrategyDef(
        key="sma_cross",
        name="SMA crossover",
        style="trend",
        description=(
            "Long while the fast SMA is above the slow SMA; short stance shorts "
            "while it is below. Classic trend-following crossover."
        ),
        param_grid={"fast": [10, 20], "slow": [50, 100]},
        make_signal=_sma_cross_signal,
        levels=_sma_cross_levels,
    )
)
