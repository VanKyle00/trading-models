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

from tradinglib.features.technical import rsi, sma
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


def _rsi2_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels:
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


def _macd_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels:
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


def _bollinger_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels:
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
