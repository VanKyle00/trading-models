"""Breakout tournament strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.features.technical import atr, pct_off_52w_high, range_tightness, volume_ratio
from tradinglib.tournament.levels import Levels, direction, two_r_target
from tradinglib.tournament.strategies._core import (
    StrategyDef,
    _full_history,
    _hold_between,
    publish_levels,
    register,
    score_orders,
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


def _donchian_resting_levels(bars: pd.DataFrame, params: dict, stance: str) -> pd.DataFrame:
    """Per-bar resting buy/sell-stop at the N-day channel extreme, stop at mid.

    Through-today (UNshifted); ``levels`` reads the last row, ``resting_orders``
    reads it shifted one bar. One series => publish and score cannot drift.
    """
    n = params["n"]
    upper = bars["high"].rolling(n).max()
    lower = bars["low"].rolling(n).min()
    mid = (upper + lower) / 2.0
    entry = upper if direction(stance) > 0 else lower
    return pd.DataFrame(
        {
            "entry": entry,
            "stop": mid,
            "target": two_r_target(entry, mid),
            "entry_type": "stop",
            # degenerate channel -> no stop distance, no actionable entry
            "armed": (entry - mid).abs() > 0.0,
        }
    )


def _donchian_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    long_side = direction(stance) > 0
    return publish_levels(
        _donchian_resting_levels(bars, params, stance),
        condition=(
            f"{'buy' if long_side else 'sell'}-stop at the {params['n']}-day "
            f"{'high' if long_side else 'low'}; exit at mid-channel"
        ),
    )


def _donchian_resting_orders(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.DataFrame:
    return score_orders(_donchian_resting_levels(_full_history(train, test), params, stance), test)


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
        resting_orders=_donchian_resting_orders,
    )
)

# --- base_breakout ------------------------------------------------------------

_NEAR_EXTREME = 0.15  # within 15% of the 52-week extreme
_TIGHT_Q = 0.30
_VOL_DRY = 1.0


def _base_breakout_state(
    full: pd.DataFrame, n: int, stance: str
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close, high, low, volume = full["close"], full["high"], full["low"], full["volume"]
    upper = high.rolling(n).max().shift(1)
    lower = low.rolling(n).min().shift(1)
    mid = (upper + lower) / 2.0
    tightness = range_tightness(high, low, close, n)
    tight = tightness < tightness.rolling(252).quantile(_TIGHT_Q)
    vol_dry = volume_ratio(volume, 50).rolling(10).mean() < _VOL_DRY
    if direction(stance) > 0:
        near = pct_off_52w_high(close, 252) >= -_NEAR_EXTREME
    else:
        near = close / low.rolling(252).min() - 1.0 <= _NEAR_EXTREME
    return upper, lower, mid, (tight & vol_dry & near)


def _base_breakout_signal(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.Series:
    full = _full_history(train, test)
    upper, lower, mid, setup = _base_breakout_state(full, params["base"], stance)
    armed = setup.shift(1, fill_value=False)  # base formed BEFORE the breakout bar
    close = full["close"]
    if direction(stance) > 0:
        pos = _hold_between((close > upper) & armed, close < mid)
    else:
        pos = -_hold_between((close < lower) & armed, close > mid)
    return pos.loc[test.index]


def _base_breakout_resting_levels(bars: pd.DataFrame, params: dict, stance: str) -> pd.DataFrame:
    """Per-bar resting stop over the N-bar base extreme, armed only while a tight,
    dry-volume base near the 52-week extreme is in force (through-today, UNshifted).

    Note the entry rests at the through-today rolling extreme (``.iloc[-1]`` of
    ``rolling(n).max()``), exactly the level the issued ticket rests on — NOT the
    ``shift(1)`` extreme the position-series signal crosses (audit A2: that skew
    is precisely the cert-vs-ticket mismatch this path removes).
    """
    n = params["base"]
    _, _, _, setup = _base_breakout_state(bars, n, stance)
    long_side = direction(stance) > 0
    base_high = bars["high"].rolling(n).max()
    base_low = bars["low"].rolling(n).min()
    entry = base_high if long_side else base_low
    proximate = bars["close"] >= base_high * 0.95 if long_side else bars["close"] <= base_low * 1.05
    last_atr = atr(bars["high"], bars["low"], bars["close"], 14)
    if long_side:
        stop = np.maximum(base_low, entry - 2.0 * last_atr)
    else:
        stop = np.minimum(base_high, entry + 2.0 * last_atr)
    return pd.DataFrame(
        {
            "entry": entry,
            "stop": stop,
            "target": two_r_target(entry, stop),
            "entry_type": "stop",
            "armed": setup & proximate & ((entry - stop).abs() > 0.0),
        }
    )


def _base_breakout_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    long_side = direction(stance) > 0
    return publish_levels(
        _base_breakout_resting_levels(bars, params, stance),
        condition=(
            f"{'buy' if long_side else 'sell'}-stop over the {params['base']}-bar base "
            f"{'high' if long_side else 'low'} after a tight, dry-volume base "
            f"near the 52-week {'high' if long_side else 'low'}"
        ),
    )


def _base_breakout_resting_orders(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.DataFrame:
    frame = _base_breakout_resting_levels(_full_history(train, test), params, stance)
    return score_orders(frame, test)


register(
    StrategyDef(
        key="base_breakout",
        name="Base breakout",
        style="breakout",
        description=(
            "Tight consolidation near the 52-week extreme on drying volume; "
            "enter on the break of the base boundary, exit at mid-base."
        ),
        param_grid={"base": [30, 40, 55]},
        make_signal=_base_breakout_signal,
        levels=_base_breakout_levels,
        resting_orders=_base_breakout_resting_orders,
    )
)
