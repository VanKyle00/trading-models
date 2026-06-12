"""Stage 2: swing-setup detectors for the 2-week-to-6-month horizon.

Each detector inspects the LAST bar of a daily OHLCV frame (the
``load_daily`` canonical schema: lowercase columns plus ``symbol``) and
returns a :class:`SetupSignal` when the setup is forming now, else ``None``.
Three setups for v1:

- ``base_breakout``: tight consolidation near the 52-week high on drying-up
  volume — entry is the breakout over the base high.
- ``ma_pullback``: orderly pullback to a rising 50-day MA inside an uptrend.
- ``pead``: post-earnings-announcement drift after a big up-gap on volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from tradinglib.features.technical import (
    atr,
    ma_slope,
    pct_off_52w_high,
    range_tightness,
    relative_strength,
    rsi,
    sma,
    volume_ratio,
)

_MIN_BARS = 260  # ~1 trading year + headroom for trailing windows
_BASE_WINDOW = 40
_NEAR_HIGH = -0.15
_NEAR_LOW = 0.15
_BREAKOUT_PROXIMITY = 0.05
_TIGHTNESS_QUANTILE = 0.30
_PULLBACK_BAND = 0.025
_LEG_MIN = 0.15
_PEAD_MIN_MOVE = 0.04
_PEAD_MIN_VOLUME_RATIO = 1.5
_PEAD_WINDOW = (3, 15)  # trading days since the earnings reaction


@dataclass
class SetupSignal:
    """One detected setup on one ticker, as of the last bar."""

    ticker: str
    setup_type: str
    score: float  # 0-1, comparable across tickers within a setup type
    asof: str
    trigger_level: float
    stop_level: float
    evidence: dict = field(default_factory=dict)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _ticker(bars: pd.DataFrame) -> str:
    return str(bars["symbol"].iloc[-1]) if "symbol" in bars.columns else "?"


def _asof(bars: pd.DataFrame) -> str:
    return str(pd.Timestamp(bars.index[-1]).date())


def _rs_component(close: pd.Series, benchmark_close: pd.Series | None) -> float:
    """6-month relative strength mapped to 0-1; neutral 0.5 without a benchmark."""
    if benchmark_close is None:
        return 0.5
    rs = relative_strength(close, benchmark_close, 126).iloc[-1]
    if np.isnan(rs):
        return 0.5
    return _clip01(0.5 + rs * 2.5)


def detect_base_breakout(
    bars: pd.DataFrame, *, benchmark_close: pd.Series | None = None
) -> SetupSignal | None:
    if len(bars) < _MIN_BARS:
        return None
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]

    off_high = pct_off_52w_high(close, 252).iloc[-1]
    if not off_high >= _NEAR_HIGH:
        return None

    tightness = range_tightness(high, low, close, _BASE_WINDOW)
    threshold = tightness.iloc[-252:].quantile(_TIGHTNESS_QUANTILE)
    if not tightness.iloc[-1] < threshold:
        return None

    vol_ratio_10 = volume_ratio(volume, 50).iloc[-10:].mean()
    if not vol_ratio_10 < 1.0:
        return None

    base_high = high.iloc[-_BASE_WINDOW:].max()
    base_low = low.iloc[-_BASE_WINDOW:].min()
    if not close.iloc[-1] >= base_high * (1.0 - _BREAKOUT_PROXIMITY):
        return None

    last_atr = atr(high, low, close, 14).iloc[-1]
    tightness_pct = float((tightness.iloc[-252:] <= tightness.iloc[-1]).mean())
    proximity = 1.0 - min(abs(off_high) / abs(_NEAR_HIGH), 1.0)
    score = _clip01(
        0.4 * (1.0 - tightness_pct) + 0.3 * proximity + 0.3 * _rs_component(close, benchmark_close)
    )
    return SetupSignal(
        ticker=_ticker(bars),
        setup_type="base_breakout",
        score=score,
        asof=_asof(bars),
        trigger_level=float(base_high),
        stop_level=float(max(base_low, close.iloc[-1] - 2.0 * last_atr)),
        evidence={
            "pct_off_52w_high": float(off_high),
            "range_tightness": float(tightness.iloc[-1]),
            "volume_ratio_10d": float(vol_ratio_10),
            "base_high": float(base_high),
            "base_low": float(base_low),
            "atr": float(last_atr),
        },
    )


def detect_base_breakdown(
    bars: pd.DataFrame, *, benchmark_close: pd.Series | None = None
) -> SetupSignal | None:
    if len(bars) < _MIN_BARS:
        return None
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]

    low_252 = close.iloc[-252:].min()
    off_low = close.iloc[-1] / low_252 - 1.0  # distance ABOVE the 52-week low
    if not off_low <= _NEAR_LOW:
        return None

    tightness = range_tightness(high, low, close, _BASE_WINDOW)
    threshold = tightness.iloc[-252:].quantile(_TIGHTNESS_QUANTILE)
    if not tightness.iloc[-1] < threshold:
        return None

    vol_ratio_10 = volume_ratio(volume, 50).iloc[-10:].mean()
    if not vol_ratio_10 < 1.0:
        return None

    base_high = high.iloc[-_BASE_WINDOW:].max()
    base_low = low.iloc[-_BASE_WINDOW:].min()
    if not close.iloc[-1] <= base_low * (1.0 + _BREAKOUT_PROXIMITY):
        return None

    last_atr = atr(high, low, close, 14).iloc[-1]
    tightness_pct = float((tightness.iloc[-252:] <= tightness.iloc[-1]).mean())
    proximity = 1.0 - min(off_low / _NEAR_LOW, 1.0)
    # weak relative strength is the FAVORABLE condition for a short
    score = _clip01(
        0.4 * (1.0 - tightness_pct)
        + 0.3 * proximity
        + 0.3 * (1.0 - _rs_component(close, benchmark_close))
    )
    return SetupSignal(
        ticker=_ticker(bars),
        setup_type="base_breakdown",
        score=score,
        asof=_asof(bars),
        trigger_level=float(base_low),
        stop_level=float(min(base_high, close.iloc[-1] + 2.0 * last_atr)),
        evidence={
            "pct_off_52w_low": float(off_low),
            "range_tightness": float(tightness.iloc[-1]),
            "volume_ratio_10d": float(vol_ratio_10),
            "base_high": float(base_high),
            "base_low": float(base_low),
            "atr": float(last_atr),
        },
    )


def detect_ma_pullback(
    bars: pd.DataFrame, *, benchmark_close: pd.Series | None = None
) -> SetupSignal | None:
    if len(bars) < _MIN_BARS:
        return None
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]

    sma50 = sma(close, 50).iloc[-1]
    sma200 = sma(close, 200).iloc[-1]
    if not (sma50 > sma200):
        return None
    if not (ma_slope(close, 50, 10).iloc[-1] > 0 and ma_slope(close, 200, 20).iloc[-1] > 0):
        return None

    leg = close.iloc[-60:].max() / close.iloc[-120:].min() - 1.0
    if not leg >= _LEG_MIN:
        return None

    high_60 = close.iloc[-60:].max()
    pulled_back = close.iloc[-1] <= high_60 * 0.95
    at_the_ma = abs(close.iloc[-1] / sma50 - 1.0) <= _PULLBACK_BAND
    if not (pulled_back and at_the_ma):
        return None

    last_rsi = rsi(close, 14).iloc[-1]
    if not (30.0 <= last_rsi <= 50.0):
        return None

    last_atr = atr(high, low, close, 14).iloc[-1]
    vol_ratio_10 = volume_ratio(volume, 50).iloc[-10:].mean()
    trend = _clip01((sma50 / sma200 - 1.0) * 10.0)
    orderliness = _clip01(2.0 - vol_ratio_10) if vol_ratio_10 >= 1.0 else 1.0
    score = _clip01(0.4 * trend + 0.3 * orderliness + 0.3 * _rs_component(close, benchmark_close))
    return SetupSignal(
        ticker=_ticker(bars),
        setup_type="ma_pullback",
        score=score,
        asof=_asof(bars),
        trigger_level=float(sma(close, 20).iloc[-1]),
        stop_level=float(min(low.iloc[-10:].min(), sma50 - 1.5 * last_atr)),
        evidence={
            "sma50": float(sma50),
            "sma200": float(sma200),
            "leg_gain": float(leg),
            "pullback_from_high": float(close.iloc[-1] / high_60 - 1.0),
            "rsi": float(last_rsi),
            "volume_ratio_10d": float(vol_ratio_10),
        },
    )


def detect_pead(
    bars: pd.DataFrame, earnings_datetimes: list[pd.Timestamp] | pd.DatetimeIndex
) -> SetupSignal | None:
    if len(bars) < 60 or len(earnings_datetimes) == 0:
        return None
    close, low, volume = bars["close"], bars["low"], bars["volume"]
    index = cast(pd.DatetimeIndex, bars.index)

    past = [ts for ts in pd.DatetimeIndex(earnings_datetimes) if ts <= index[-1]]
    if not past:
        return None
    # the reaction bar is the first session on/after the most recent earnings
    reaction_pos = int(index.searchsorted(max(past)))
    if reaction_pos >= len(index):
        return None
    days_since = len(index) - 1 - reaction_pos
    if not (_PEAD_WINDOW[0] <= days_since <= _PEAD_WINDOW[1]):
        return None
    if reaction_pos == 0:
        return None

    move = close.iloc[reaction_pos] / close.iloc[reaction_pos - 1] - 1.0
    if not move >= _PEAD_MIN_MOVE:
        return None
    vol_ratio = volume_ratio(volume, 50).iloc[reaction_pos]
    if not vol_ratio >= _PEAD_MIN_VOLUME_RATIO:
        return None
    reaction_low = low.iloc[reaction_pos]
    if not close.iloc[-1] > reaction_low:
        return None

    reaction_close = close.iloc[reaction_pos]
    if close.iloc[-1] >= reaction_close:
        hold = 1.0
    else:
        hold = _clip01(
            1.0 - (reaction_close - close.iloc[-1]) / max(reaction_close - reaction_low, 1e-9)
        )
    score = _clip01(
        0.5 * min(move / 0.10, 1.0) + 0.3 * min(float(vol_ratio) / 3.0, 1.0) + 0.2 * hold
    )
    return SetupSignal(
        ticker=_ticker(bars),
        setup_type="pead",
        score=score,
        asof=_asof(bars),
        trigger_level=float(close.iloc[-1]),
        stop_level=float(reaction_low),
        evidence={
            "earnings_day_move": float(move),
            "earnings_day_volume_ratio": float(vol_ratio),
            "days_since_earnings": days_since,
            "reaction_low": float(reaction_low),
        },
    )


def detect_all(
    bars: pd.DataFrame,
    *,
    benchmark_close: pd.Series | None = None,
    earnings_datetimes: list[pd.Timestamp] | pd.DatetimeIndex | None = None,
) -> list[SetupSignal]:
    """Run every detector on one ticker's bars; return all matches."""
    signals = [
        detect_base_breakout(bars, benchmark_close=benchmark_close),
        detect_ma_pullback(bars, benchmark_close=benchmark_close),
    ]
    if earnings_datetimes is not None:
        signals.append(detect_pead(bars, earnings_datetimes))
    return [s for s in signals if s is not None]
