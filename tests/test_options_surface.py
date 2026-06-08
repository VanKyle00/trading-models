"""Tests for the synthetic implied-volatility surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.options.surface import (
    FlatSurface,
    ParametricSurface,
    realistic_surface,
    realized_vol,
)


def _prices(n: int = 200, vol: float = 0.2, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rets = rng.normal(0.0, vol / np.sqrt(252), n)
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)


def test_flat_surface_is_constant() -> None:
    s = FlatSurface(0.18)
    t = pd.Timestamp("2024-01-01")
    assert s.iv(100.0, 90.0, t + pd.Timedelta(days=30), t) == 0.18
    assert s.iv(100.0, 110.0, t + pd.Timedelta(days=180), t) == 0.18


def test_realized_vol_recovers_input_scale() -> None:
    rv = realized_vol(_prices(n=2000, vol=0.2, seed=1), window=21).dropna()
    assert 0.17 < rv.mean() < 0.23


def test_parametric_surface_has_equity_skew() -> None:
    atm = pd.Series(0.20, index=pd.date_range("2024-01-01", periods=5, freq="B"))
    s = ParametricSurface(atm_vol=atm)
    t = atm.index[0]
    expiry = t + pd.Timedelta(days=60)
    assert (
        s.iv(100.0, 90.0, expiry, t) > s.iv(100.0, 100.0, expiry, t) > s.iv(100.0, 110.0, expiry, t)
    )


def test_parametric_atm_equals_input_at_reference_window() -> None:
    t0 = pd.Timestamp("2024-01-01")
    atm = pd.Series([0.15, 0.40], index=[t0, t0 + pd.Timedelta(days=10)])
    s = ParametricSurface(atm_vol=atm)
    # ATM (m=0) at the 21-day reference window: term_factor == 1, skew == 1.
    low = s.iv(100.0, 100.0, t0 + pd.Timedelta(days=21), t0)
    assert low == pytest.approx(0.15, abs=1e-9)
    high = s.iv(100.0, 100.0, t0 + pd.Timedelta(days=70), t0 + pd.Timedelta(days=10))
    assert high > 0.35  # tracks the 0.40 ATM input, not the 0.15 earlier regime


def test_long_dated_skew_is_flatter() -> None:
    atm = pd.Series(0.20, index=pd.date_range("2024-01-01", periods=5, freq="B"))
    s = ParametricSurface(atm_vol=atm)
    t = atm.index[0]
    short = s.iv(100.0, 90.0, t + pd.Timedelta(days=30), t) - s.iv(
        100.0, 110.0, t + pd.Timedelta(days=30), t
    )
    long = s.iv(100.0, 90.0, t + pd.Timedelta(days=300), t) - s.iv(
        100.0, 110.0, t + pd.Timedelta(days=300), t
    )
    assert short > long > 0


def test_realistic_surface_atm_is_time_varying_and_clipped() -> None:
    prices = _prices(n=200, vol=0.2, seed=2)
    s = realistic_surface(prices, vrp=1.15)
    assert isinstance(s, ParametricSurface)
    t_early, t_late = prices.index[40], prices.index[-1]
    iv_early = s.iv(100.0, 100.0, t_early + pd.Timedelta(days=60), t_early)
    iv_late = s.iv(100.0, 100.0, t_late + pd.Timedelta(days=60), t_late)
    assert iv_early != iv_late
    assert 0.02 <= iv_late <= 3.0
