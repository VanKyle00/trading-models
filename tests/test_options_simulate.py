"""Tests for GBM path simulation and the strategy outcome simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tradinglib.options.simulate import gbm_paths


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
