"""Synthetic implied-volatility surface for the options engine.

No real options quotes — the surface is *calibrated to the underlying's own
realized volatility* and overlaid with parametric skew and term structure. It is
a stress / plausibility model: it reproduces the *shape* of real frictions (vol
regimes, skew, term structure), not the exact IV of any contract.

IV is a separable product::

    IV(K, expiry, t) = atm_vol(t) * term_factor(dte) * skew_factor(m, dte)

with ``dte = (expiry - t).days`` and ``m = log(K / spot)`` (log-moneyness).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class VolSurface(Protocol):
    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float: ...


@dataclass(frozen=True)
class FlatSurface:
    """Constant IV everywhere — reproduces the pre-surface engine behavior."""

    vol: float

    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
        return self.vol


@dataclass(frozen=True)
class SurfaceParams:
    """Shape parameters for :class:`ParametricSurface` (equity-index-typical)."""

    skew_slope: float = -0.30  # b: <0 => OTM puts richer than OTM calls
    skew_curv: float = 0.50  # c: smile curvature (>= 0)
    skew_flatten: float = 2.0  # k: skew slope decays with tenor (per year)
    term_slope: float = 0.05  # term-structure slope per sqrt-year
    ref_window_days: int = 21  # dte at which term_factor == 1.0
    iv_floor: float = 0.02
    iv_cap: float = 3.0


def realized_vol(prices: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Trailing annualized realized vol from close-to-close log returns."""
    logret = np.log(prices / prices.shift(1))
    return logret.rolling(window).std() * math.sqrt(periods_per_year)


@dataclass
class ParametricSurface:
    """Realized-vol-anchored surface with parametric skew + term structure.

    Not frozen because ``atm_vol`` is a mutable ``pd.Series`` (sorted in ``__post_init__``).
    """

    atm_vol: pd.Series  # time-indexed ATM vol (annualized)
    params: SurfaceParams = field(default_factory=SurfaceParams)

    def __post_init__(self) -> None:
        if not self.atm_vol.index.is_monotonic_increasing:
            self.atm_vol = self.atm_vol.sort_index()

    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
        p = self.params
        atm = float(self.atm_vol.asof(t))
        if not math.isfinite(atm):
            # NaN from asof means t is before the series start; use the earliest
            # available ATM value as a pre-history estimate.
            valid = self.atm_vol.dropna()
            atm = float(valid.iloc[0]) if not valid.empty else p.iv_floor
        years = max((expiry - t).days, 0) / 365.0
        m = math.log(strike / spot) if spot > 0 and strike > 0 else 0.0

        b_eff = p.skew_slope / (1.0 + p.skew_flatten * years)
        skew = 1.0 + b_eff * m + p.skew_curv * m * m
        term = 1.0 + p.term_slope * (math.sqrt(years) - math.sqrt(p.ref_window_days / 365.0))

        iv = atm * max(term, 0.0) * max(skew, 0.0)
        return float(min(max(iv, p.iv_floor), p.iv_cap))


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Drop tz so naive (engine bars) and aware (loader earnings) compare cleanly."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


@dataclass(frozen=True)
class EventVolSurface:
    """Synthetic two-regime IV for the Phase-1 earnings straddle.

    Returns ``pre_iv`` (elevated) on bars at or before the earnings datetime and
    ``post_iv`` (crushed) strictly after it. Implements the ``VolSurface``
    protocol. Enforces ``pre_iv > post_iv > 0`` because IV crush is the entire
    point: a config with ``post_iv >= pre_iv`` would manufacture an IV expansion
    into the move and make the long straddle falsely profitable. Phase-1 only —
    clearly not tradeable, mirrors the repo's synthetic vol treatment. tz-robust:
    coerces both operands to tz-naive so a UTC-aware loader earnings timestamp
    compares cleanly with tz-naive engine bars.
    """

    earnings_datetime: pd.Timestamp
    pre_iv: float
    post_iv: float

    def __post_init__(self) -> None:
        if not (self.pre_iv > self.post_iv > 0):
            raise ValueError(
                f"require pre_iv > post_iv > 0 (IV crush); got pre_iv={self.pre_iv}, "
                f"post_iv={self.post_iv}"
            )

    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
        return self.pre_iv if _to_naive(t) <= _to_naive(self.earnings_datetime) else self.post_iv


def realistic_surface(
    prices: pd.Series,
    *,
    window: int = 21,
    vrp: float = 1.15,
    periods_per_year: int = 252,
    params: SurfaceParams | None = None,
) -> ParametricSurface:
    """Build a :class:`ParametricSurface` with ``atm_vol = realized_vol * vrp``.

    Leading NaNs from the rolling window are back-filled so early bars are usable.
    """
    atm = realized_vol(prices, window=window, periods_per_year=periods_per_year) * vrp
    atm = atm.bfill()
    if atm.isna().all():
        warnings.warn(
            f"realized vol is all-NaN (need > window={window} bars); "
            "ATM vol defaults to the floor everywhere",
            stacklevel=2,
        )
    return ParametricSurface(atm_vol=atm, params=params or SurfaceParams())
