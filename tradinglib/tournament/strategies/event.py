"""Event-driven tournament strategies (post-earnings-announcement drift).

Bars MAY carry a boolean ``earnings`` column (True on the reaction bar — the
session that reacts to an announcement: a before-open report's own session, an
after-close/undated report's next session; the scan pipeline attaches it,
session-aware per #91/#96). When the column is absent the signal is all-flat and
levels are ``None``, so the strategy simply never survives in contexts that lack
event data.
"""

from __future__ import annotations

import pandas as pd

from tradinglib.features.technical import volume_ratio
from tradinglib.tournament.levels import Levels, direction, two_r_target
from tradinglib.tournament.strategies._core import (
    StrategyDef,
    _full_history,
    _hold_between,
    register,
)

_ENTRY_DELAY = 3  # sessions after the reaction bar (start of the PEAD window)
_MIN_VOLUME_RATIO = 1.5


def _reactions(full: pd.DataFrame, min_move: float, stance: str) -> pd.Series:
    move = full["close"].pct_change()
    vr = volume_ratio(full["volume"], 50)
    qualified = vr >= _MIN_VOLUME_RATIO
    if direction(stance) > 0:
        qualified &= move >= min_move
    else:
        qualified &= move <= -min_move
    return full["earnings"].eq(True) & qualified


def _pead_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str) -> pd.Series:
    full = _full_history(train, test)
    if "earnings" not in full.columns:
        return pd.Series(0.0, index=test.index)
    reaction = _reactions(full, params["min_move"], stance)
    close = full["close"]
    # Overlapping reactions: _hold_between makes a second reaction's entry a
    # no-op while in position, and the FIRST reaction's time-exit still fires
    # (a later overlapping window may be partially missed); the ffill'd guard
    # jumps to the newest reaction's extreme. Back-to-back qualifying earnings
    # are rare; first-exit-wins is the accepted semantics.
    if direction(stance) > 0:
        guard = full["low"].where(reaction).ffill()
        entries = reaction.shift(_ENTRY_DELAY, fill_value=False) & (close > guard)
        exits = reaction.shift(_ENTRY_DELAY + params["hold"], fill_value=False) | (close < guard)
        pos = _hold_between(entries, exits)
    else:
        guard = full["high"].where(reaction).ffill()
        entries = reaction.shift(_ENTRY_DELAY, fill_value=False) & (close < guard)
        exits = reaction.shift(_ENTRY_DELAY + params["hold"], fill_value=False) | (close > guard)
        pos = -_hold_between(entries, exits)
    return pos.loc[test.index]


def _pead_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    if "earnings" not in bars.columns:
        return None
    reaction = _reactions(bars, params["min_move"], stance)
    if not bool(reaction.any()):
        return None
    last_reaction = int(reaction.to_numpy().nonzero()[0][-1])
    days_since = len(bars) - 1 - last_reaction
    if not (_ENTRY_DELAY <= days_since < _ENTRY_DELAY + params["hold"]):
        return None  # outside the drift window tonight (upper bound = the signal's exit bar)
    long_side = direction(stance) > 0
    entry = float(bars["close"].iloc[-1])
    stop = float(bars["low"].iloc[last_reaction] if long_side else bars["high"].iloc[last_reaction])
    if not direction(stance) * (entry - stop) > 0:
        return None  # price already through the reaction extreme
    return Levels(
        entry=entry,
        entry_type="market",
        stop=stop,
        target=two_r_target(entry, stop),
        condition=(
            f"ride post-earnings drift ({days_since} sessions after a "
            f">={params['min_move']:.0%} reaction on volume); stop at the reaction "
            f"{'low' if long_side else 'high'}"
        ),
    )


register(
    StrategyDef(
        key="pead",
        name="Post-earnings drift",
        style="event",
        description=(
            "After an earnings reaction bar moving at least min_move on 1.5x volume, "
            "hold in the drift direction from 3 sessions post-reaction for hold "
            "sessions, or out on a close through the reaction extreme."
        ),
        param_grid={"min_move": [0.04, 0.06], "hold": [7, 12]},
        make_signal=_pead_signal,
        levels=_pead_levels,
    )
)
