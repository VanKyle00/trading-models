"""Trend-following tournament strategies."""

from __future__ import annotations

import pandas as pd

from tradinglib.features.technical import sma
from tradinglib.tournament.levels import Levels, direction, protective_stop, two_r_target
from tradinglib.tournament.strategies._core import StrategyDef, _full_history, register

# --- sma_cross ---------------------------------------------------------------


def _sma_cross_signal(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.Series:
    closes = _full_history(train, test)["close"]
    fast = sma(closes, params["fast"])
    slow = sma(closes, params["slow"])
    pos = (fast > slow).astype(float) if direction(stance) > 0 else -(fast < slow).astype(float)
    return pos.loc[test.index]


def _sma_cross_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
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


# --- macd --------------------------------------------------------------------


def _macd_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str) -> pd.Series:
    close = _full_history(train, test)["close"]
    macd_line = (
        close.ewm(span=params["fast"], adjust=False).mean()
        - close.ewm(span=params["slow"], adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=params["signal"], adjust=False).mean()
    if direction(stance) > 0:
        pos = (macd_line > signal_line).astype(float)
    else:
        pos = -(macd_line < signal_line).astype(float)
    return pos.loc[test.index]


def _macd_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    entry = float(bars["close"].iloc[-1])
    stop = protective_stop(bars, entry, stance)
    side = "above" if direction(stance) > 0 else "below"
    return Levels(
        entry=entry,
        entry_type="market",
        stop=stop,
        target=two_r_target(entry, stop),
        condition=(
            f"enter while MACD({params['fast']},{params['slow']}) holds {side} its "
            f"{params['signal']}-bar signal line; rule exits on the opposite cross"
        ),
    )


register(
    StrategyDef(
        key="macd",
        name="MACD cross",
        style="trend",
        description=(
            "Long while the MACD line is above its signal line; short stance "
            "shorts while it is below."
        ),
        param_grid={"fast": [8, 12], "slow": [17, 26], "signal": [9]},
        make_signal=_macd_signal,
        levels=_macd_levels,
    )
)
