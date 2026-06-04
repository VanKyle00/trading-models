"""Tests for GBM path simulation and the strategy outcome simulation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest.options_engine import OptionsEngine
from tradinglib.options.instruments import OptionLeg
from tradinglib.options.simulate import SimulationResult, gbm_paths, run_simulation


def test_paths_shape_and_start() -> None:
    paths = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=252, n_paths=1000, seed=0)
    assert paths.shape == (1000, 253)  # days + 1 (includes the t=0 column)
    assert np.allclose(paths[:, 0], 100.0)


def test_paths_are_float32() -> None:
    paths = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=10, n_paths=10, seed=0)
    assert paths.dtype == np.float32


def test_terminal_mean_matches_risk_neutral_drift() -> None:
    spot, vol, rate, days, n = 100.0, 0.2, 0.05, 252, 200_000
    paths = gbm_paths(spot=spot, vol=vol, rate=rate, days=days, n_paths=n, seed=42)
    terminal_mean = float(paths[:, -1].mean())
    t_years = days / 252
    expected = spot * math.exp(rate * t_years)
    assert terminal_mean == pytest.approx(expected, rel=0.01)


def test_seed_is_deterministic() -> None:
    a = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=20, n_paths=50, seed=7)
    b = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=20, n_paths=50, seed=7)
    assert np.array_equal(a, b)


def _long_call_factory(expiry_index: int):
    """Return a strategy-factory that buys one ATM call at bar 0."""

    def factory():
        class BuyCall:
            def __init__(self) -> None:
                self.opened = False
                self.expiry_index = expiry_index

            def on_bar(self, engine: OptionsEngine, t, spot) -> None:
                if not self.opened:
                    expiry = t + pd.Timedelta(days=30)
                    engine.add_leg(OptionLeg("call", strike=round(spot), expiry=expiry, quantity=1.0))
                    self.opened = True

        return BuyCall()

    return factory


def test_run_simulation_returns_distribution() -> None:
    result = run_simulation(
        _long_call_factory(20),
        spot=100.0,
        vol=0.2,
        rate=0.04,
        days=20,
        n_paths=500,
        seed=1,
    )
    assert isinstance(result, SimulationResult)
    assert result.pnl_distribution.shape == (500,)
    assert 0.0 <= result.prob_of_profit <= 1.0
    assert set(result.percentiles) == {5, 25, 50, 75, 95}
    assert result.percentiles[5] <= result.percentiles[95]


def test_simulation_respects_max_paths_cap() -> None:
    result = run_simulation(
        _long_call_factory(10),
        spot=100.0,
        vol=0.2,
        rate=0.04,
        days=10,
        n_paths=10_000,
        max_paths=1_000,
        seed=2,
    )
    assert result.pnl_distribution.shape == (1_000,)
    assert result.truncated is True
