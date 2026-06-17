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

from tradinglib.tournament.levels import Levels

# (train, test, params, stance) -> target-position series indexed like test
TournamentSignalFn = Callable[[pd.DataFrame, pd.DataFrame, dict, str], pd.Series]
# (bars, params, stance) -> Levels for tomorrow, or None when not actionable tonight
LevelsFn = Callable[[pd.DataFrame, dict, str], Levels | None]
# (train, test, params, stance) -> per-bar resting-order frame indexed like test,
# columns entry/stop/target/entry_type/armed; levels are shift(1)-causal so the
# order resting on bar t is the one published after bar t-1 (audit A2). Set on a
# StrategyDef only when the tournament should score the issued resting order
# rather than a {0, ±1} position series; None keeps the position-series path.
RestingOrdersFn = Callable[[pd.DataFrame, pd.DataFrame, dict, str], pd.DataFrame]

ORDER_COLUMNS = ["entry", "stop", "target", "entry_type", "armed"]


@dataclass(frozen=True)
class StrategyDef:
    key: str
    name: str
    style: str  # "trend" | "breakout" | "mean_reversion" | "event" | "ml"
    description: str  # plain-English rule; powers the /models page
    param_grid: dict[str, list]
    make_signal: TournamentSignalFn
    levels: LevelsFn
    # Purged-CV embargo: bars to drop from the end of each walk-forward train
    # window so a label looking this many bars ahead cannot bleed across the
    # train/test boundary. 0 = the strategy is causal / self-purging (every
    # shipped strategy); a future model with a forward-looking H-bar label sets
    # cv_embargo=H to have the CV layer enforce the purge automatically.
    cv_embargo: int = 0
    # Resting-order builder for the intrabar tournament scoring path (audit A2).
    # None (the default) => the strategy is scored on the position-series path
    # exactly as before. Set it (donchian/base_breakout/bollinger) so the
    # tournament's resting engine scores the SAME stop/limit order the ticket
    # issues. Pairs with ``levels``: both derive from one ``_resting_levels``
    # series, publish-unshifted / score-shifted, so they cannot drift.
    resting_orders: RestingOrdersFn | None = None


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


def publish_levels(frame: pd.DataFrame, condition: str) -> Levels | None:
    """Tonight's published order = the last (through-today, UNshifted) row of a
    per-bar resting-levels frame. ``None`` when that bar is not armed — the same
    "no actionable entry tonight" answer the hand-written ``levels`` returned.
    """
    row = frame.iloc[-1]
    if not bool(row["armed"]):
        return None
    return Levels(
        entry=float(row["entry"]),
        entry_type=str(row["entry_type"]),
        stop=float(row["stop"]),
        target=float(row["target"]),
        condition=condition,
    )


def score_orders(frame: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """The resting orders for the test slice = the per-bar levels frame SHIFTED
    one bar (the order resting on bar t is the one published after bar t-1), then
    restricted to ``test``. Disarmed wherever the prior bar published nothing.
    """
    orders = pd.DataFrame(index=frame.index)
    orders["entry"] = frame["entry"].shift(1)
    orders["stop"] = frame["stop"].shift(1)
    orders["target"] = frame["target"].shift(1)
    orders["entry_type"] = frame["entry_type"]  # constant per strategy
    orders["armed"] = frame["armed"].shift(1, fill_value=False).astype(bool)
    return orders.loc[test.index, ORDER_COLUMNS].copy()
