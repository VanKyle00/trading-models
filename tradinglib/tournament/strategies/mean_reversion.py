"""Mean-reversion tournament strategies."""

from __future__ import annotations

import pandas as pd

from tradinglib.features.technical import atr, ma_slope, rsi, sma
from tradinglib.tournament.levels import (
    _ATR_MULT,
    _ATR_WINDOW,
    Levels,
    direction,
    protective_stop,
    two_r_target,
)
from tradinglib.tournament.strategies._core import (
    StrategyDef,
    _full_history,
    _hold_between,
    publish_levels,
    register,
    score_orders,
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


def _bollinger_resting_levels(bars: pd.DataFrame, params: dict, stance: str) -> pd.DataFrame:
    """Per-bar resting limit at the band, target at the mean, 2x ATR(14) stop
    (through-today, UNshifted).

    The stop mirrors ``protective_stop`` exactly (same ``_ATR_MULT`` / ``_ATR_WINDOW``)
    so the published last-row order is byte-identical to the old ``levels``; a bar
    with a degenerate (zero/NaN) ATR — the case the scalar ``protective_stop``
    raised on — is simply not armed (``levels`` returns None there, the documented
    "no actionable entry" answer).
    """
    mid, lower, upper = _bollinger_bands(bars["close"], params["window"], params["num_std"])
    long_side = direction(stance) > 0
    entry = lower if long_side else upper
    last_atr = atr(bars["high"], bars["low"], bars["close"], _ATR_WINDOW)
    stop = entry - direction(stance) * _ATR_MULT * last_atr
    return pd.DataFrame(
        {
            "entry": entry,
            "stop": stop,
            "target": mid,
            "entry_type": "limit",
            "armed": (last_atr > 0.0) & entry.notna() & mid.notna(),
        }
    )


def _bollinger_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    long_side = direction(stance) > 0
    return publish_levels(
        _bollinger_resting_levels(bars, params, stance),
        condition=(
            f"fade to the {'lower' if long_side else 'upper'} "
            f"Bollinger({params['window']}, {params['num_std']}) band; exit at the mean"
        ),
    )


def _bollinger_resting_orders(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.DataFrame:
    return score_orders(_bollinger_resting_levels(_full_history(train, test), params, stance), test)


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
        resting_orders=_bollinger_resting_orders,
    )
)


# --- ma_pullback -------------------------------------------------------------

_LEG_MIN = 0.15
_PULLED = 0.95


def _ma_pullback_setup(full: pd.DataFrame, band: float, stance: str) -> pd.Series:
    close = full["close"]
    s50, s200 = sma(close, 50), sma(close, 200)
    r = rsi(close, 14)
    at_ma = (close / s50 - 1.0).abs() <= band
    if direction(stance) > 0:
        trend_ok = (s50 > s200) & (ma_slope(close, 50, 10) > 0) & (ma_slope(close, 200, 20) > 0)
        leg = close.rolling(60).max() / close.rolling(120).min() - 1.0 >= _LEG_MIN
        pulled = close <= close.rolling(60).max() * _PULLED
        rsi_ok = (r >= 30.0) & (r <= 50.0)
    else:
        trend_ok = (s50 < s200) & (ma_slope(close, 50, 10) < 0) & (ma_slope(close, 200, 20) < 0)
        leg = close.rolling(60).min() / close.rolling(120).max() - 1.0 <= -_LEG_MIN
        pulled = close >= close.rolling(60).min() / _PULLED
        rsi_ok = (r >= 50.0) & (r <= 70.0)
    return trend_ok & leg & pulled & at_ma & rsi_ok


def _ma_pullback_signal(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str
) -> pd.Series:
    # Enter on the SMA(20) recapture bar (setup armed on the PRIOR bar) so the
    # scored rule matches the levels' stop-entry at SMA(20); exit on losing
    # SMA(20) or on trend failure at SMA(200).
    full = _full_history(train, test)
    close = full["close"]
    setup = _ma_pullback_setup(full, params["band"], stance)
    armed = setup.shift(1, fill_value=False)
    s20, s200 = sma(close, 20), sma(close, 200)
    if direction(stance) > 0:
        pos = _hold_between(armed & (close > s20), (close < s20) | (close < s200))
    else:
        pos = -_hold_between(armed & (close < s20), (close > s20) | (close > s200))
    return pos.loc[test.index]


def _ma_pullback_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    if not bool(_ma_pullback_setup(bars, params["band"], stance).iloc[-1]):
        return None  # not at the MA in a qualifying trend tonight
    close = bars["close"]
    long_side = direction(stance) > 0
    entry = float(sma(close, 20).iloc[-1])  # recapture trigger, like the detector
    last_atr = float(atr(bars["high"], bars["low"], close, 14).iloc[-1])
    s50 = float(sma(close, 50).iloc[-1])
    if long_side:
        stop = min(float(bars["low"].iloc[-10:].min()), s50 - 1.5 * last_atr)
    else:
        stop = max(float(bars["high"].iloc[-10:].max()), s50 + 1.5 * last_atr)
    if not direction(stance) * (entry - stop) > 0:
        return None  # stop ended up on the wrong side (degenerate geometry)
    return Levels(
        entry=entry,
        entry_type="stop",
        stop=stop,
        target=two_r_target(entry, stop),
        condition=(
            f"{'buy' if long_side else 'sell'}-stop at SMA(20) recapture after an "
            f"orderly pullback to SMA(50) (band {params['band']:.0%})"
        ),
    )


register(
    StrategyDef(
        key="ma_pullback",
        name="MA pullback",
        style="mean_reversion",
        description=(
            "Orderly pullback to a rising SMA(50) inside an uptrend (RSI 30-50); "
            "enter on the SMA(20) recapture, ride while SMA(20) holds, out on trend "
            "failure at SMA(200). Short stance mirrors a downtrend rally."
        ),
        param_grid={"band": [0.02, 0.03]},
        make_signal=_ma_pullback_signal,
        levels=_ma_pullback_levels,
    )
)
