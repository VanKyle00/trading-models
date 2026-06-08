"""Per-event selection signal for the earnings straddle.

implied_move = straddle_premium / spot. This is the ATM-straddle *mean-absolute-
move* proxy (E|return|), which is ~0.8 * atm_iv * sqrt(T) — NOT the 1-sigma move
(atm_iv * sqrt(T)). expected_move below is the MEAN of past earnings-day absolute
returns (a like-for-like mean-absolute measure), so the k-gate compares two
mean-absolute quantities rather than mixing a mean against a median. Entry iff
expected_move > implied_move * k, k > 1. Plus the no-trade filters (valid IV,
post-earnings expiry, spread cap). Functions are tz-robust: they coerce earnings
datetimes and the price index to a common tz so a UTC-aware loader timestamp
compares cleanly with tz-naive bars.
"""

from __future__ import annotations

import math  # noqa: F401  used by passes_filter/tradeable_event (Task 5); pre-imported in sorted top block

import numpy as np
import pandas as pd


def implied_move(straddle_premium: float, spot: float) -> float:
    """ATM-straddle implied move = premium / spot (a fraction, e.g. 0.06).

    This is the mean-absolute-move proxy (~0.8 * iv * sqrt(T)), the like-for-like
    counterpart to the mean of past absolute earnings moves used by expected_move.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    return straddle_premium / spot


def expected_move(
    close: pd.Series,
    earnings_datetimes: pd.Series,
    lookback: int = 8,
) -> float:
    """Mean absolute earnings-day return over the last ``lookback`` events.

    For each past earnings date, the realized move is the close-to-close return
    from the earnings session to the following trading bar. Uses the MEAN (not
    median) so it is a like-for-like comparison with implied_move's mean-absolute
    proxy. tz-robust: both the price index and the earnings dates are coerced to
    tz-naive before comparison. Returns NaN if no usable history exists.
    """
    closes = close.sort_index()
    cidx = closes.index
    if getattr(cidx, "tz", None) is not None:
        cidx = cidx.tz_localize(None)
        closes = pd.Series(closes.to_numpy(), index=cidx)

    eds = pd.to_datetime(earnings_datetimes, utc=True).dt.tz_localize(None).sort_values()

    moves: list[float] = []
    for ed in eds:
        prior = closes.index[closes.index <= ed]
        if len(prior) == 0:
            continue
        pos = closes.index.get_loc(prior[-1])
        if pos + 1 >= len(closes):
            continue
        ret = closes.iloc[pos + 1] / closes.iloc[pos] - 1.0
        moves.append(abs(float(ret)))

    if not moves:
        return float("nan")
    return float(np.mean(moves[-lookback:]))
