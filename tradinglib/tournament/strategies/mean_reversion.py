"""Mean-reversion tournament strategies."""

from __future__ import annotations

import pandas as pd

from tradinglib.features.technical import rsi, sma
from tradinglib.tournament.levels import Levels, direction, protective_stop, two_r_target
from tradinglib.tournament.strategies._core import (
    StrategyDef,
    _full_history,
    _hold_between,
    register,
)

# --- rsi2 --------------------------------------------------------------------


def _rsi2_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str) -> pd.Series:
    close = _full_history(train, test)["close"]
    trend = sma(close, 200)
    fast = sma(close, 5)
    r = rsi(close, 2)
    if direction(stance) > 0:
        entries = (close > trend) & (r < params["entry_thr"])
        exits = (close > fast) | (close < trend)
        pos = _hold_between(entries, exits)
    else:
        entries = (close < trend) & (r > 100.0 - params["entry_thr"])
        exits = (close < fast) | (close > trend)
        pos = -_hold_between(entries, exits)
    return pos.loc[test.index]


def _rsi2_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    entry = float(bars["close"].iloc[-1])
    stop = protective_stop(bars, entry, stance)
    long_side = direction(stance) > 0
    thr = params["entry_thr"] if long_side else 100 - params["entry_thr"]
    return Levels(
        entry=entry,
        entry_type="market",
        stop=stop,
        target=two_r_target(entry, stop),
        condition=(
            f"enter when RSI(2) {'<' if long_side else '>'} {thr} with close "
            f"{'above' if long_side else 'below'} SMA(200); rule exits at SMA(5)"
        ),
    )


register(
    StrategyDef(
        key="rsi2",
        name="RSI(2) pullback",
        style="mean_reversion",
        description=(
            "Inside an SMA(200) trend filter, buy 2-period-RSI oversold dips and "
            "exit above SMA(5); short stance fades overbought pops in downtrends."
        ),
        param_grid={"entry_thr": [5, 10, 15]},
        make_signal=_rsi2_signal,
        levels=_rsi2_levels,
    )
)


# --- bollinger ---------------------------------------------------------------


def _bollinger_bands(
    close: pd.Series, window: int, num_std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    sd = close.rolling(window).std()
    return mid, mid - num_std * sd, mid + num_std * sd


def _bollinger_signal(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.Series:
    close = _full_history(train, test)["close"]
    mid, lower, upper = _bollinger_bands(close, params["window"], params["num_std"])
    if direction(stance) > 0:
        pos = _hold_between(close < lower, close >= mid)
    else:
        pos = -_hold_between(close > upper, close <= mid)
    return pos.loc[test.index]


def _bollinger_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    mid, lower, upper = _bollinger_bands(bars["close"], params["window"], params["num_std"])
    long_side = direction(stance) > 0
    entry = float(lower.iloc[-1]) if long_side else float(upper.iloc[-1])
    stop = protective_stop(bars, entry, stance)
    return Levels(
        entry=entry,
        entry_type="limit",
        stop=stop,
        target=float(mid.iloc[-1]),
        condition=(
            f"fade to the {'lower' if long_side else 'upper'} "
            f"Bollinger({params['window']}, {params['num_std']}) band; exit at the mean"
        ),
    )


register(
    StrategyDef(
        key="bollinger",
        name="Bollinger band fade",
        style="mean_reversion",
        description=(
            "Buy a close below the lower band and exit at the mean; short stance "
            "fades closes above the upper band back to the mean."
        ),
        param_grid={"window": [10, 20], "num_std": [2.0, 2.5]},
        make_signal=_bollinger_signal,
        levels=_bollinger_levels,
    )
)
