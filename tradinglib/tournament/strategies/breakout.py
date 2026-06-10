"""Breakout tournament strategies."""

from __future__ import annotations

import pandas as pd

from tradinglib.tournament.levels import Levels, direction, two_r_target
from tradinglib.tournament.strategies._core import (
    StrategyDef,
    _full_history,
    _hold_between,
    register,
)

# --- donchian ----------------------------------------------------------------


def _donchian_signal(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.Series:
    full = _full_history(train, test)
    n = params["n"]
    upper = full["high"].rolling(n).max().shift(1)
    lower = full["low"].rolling(n).min().shift(1)
    mid = (upper + lower) / 2.0
    if direction(stance) > 0:
        pos = _hold_between(full["close"] > upper, full["close"] < mid)
    else:
        pos = -_hold_between(full["close"] < lower, full["close"] > mid)
    return pos.loc[test.index]


def _donchian_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels:
    n = params["n"]
    upper = float(bars["high"].rolling(n).max().iloc[-1])
    lower = float(bars["low"].rolling(n).min().iloc[-1])
    mid = (upper + lower) / 2.0
    long_side = direction(stance) > 0
    entry = upper if long_side else lower
    if not abs(entry - mid) > 0:
        raise ValueError("degenerate Donchian channel; cannot place a stop")
    return Levels(
        entry=entry,
        entry_type="stop",
        stop=mid,
        target=two_r_target(entry, mid),
        condition=(
            f"{'buy' if long_side else 'sell'}-stop at the {n}-day "
            f"{'high' if long_side else 'low'}; exit at mid-channel"
        ),
    )


register(
    StrategyDef(
        key="donchian",
        name="Donchian channel breakout",
        style="breakout",
        description=(
            "Enter on a close beyond the N-day channel extreme (high for longs, "
            "low for shorts); exit when the close crosses back through mid-channel."
        ),
        param_grid={"n": [20, 40, 55]},
        make_signal=_donchian_signal,
        levels=_donchian_levels,
    )
)
