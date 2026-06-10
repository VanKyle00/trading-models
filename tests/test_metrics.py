"""Tests for Probabilistic and Deflated Sharpe ratios in compute_metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest.metrics import compute_metrics


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod() * 100_000.0


def test_new_keys_present() -> None:
    returns = pd.Series([0.01, -0.01] * 30)
    metrics = compute_metrics(returns, _equity(returns))
    assert "probabilistic_sharpe" in metrics
    assert "deflated_sharpe" in metrics


def test_zero_mean_gives_psr_half() -> None:
    # Mean per-bar return is exactly 0 -> per-bar Sharpe 0 -> PSR(0) = Phi(0) = 0.5
    returns = pd.Series([0.01, -0.01] * 40)
    metrics = compute_metrics(returns, _equity(returns), n_trials=1)
    assert metrics["probabilistic_sharpe"] == pytest.approx(0.5, abs=1e-6)
    # n_trials=1 => deflated benchmark is 0 => deflated == probabilistic
    assert metrics["deflated_sharpe"] == pytest.approx(metrics["probabilistic_sharpe"], abs=1e-9)


def test_more_trials_lowers_deflated_sharpe() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, size=250))
    base = compute_metrics(returns, _equity(returns), n_trials=1)
    many = compute_metrics(returns, _equity(returns), n_trials=100)
    # Same PSR (independent of n_trials), strictly smaller deflated Sharpe.
    assert many["probabilistic_sharpe"] == pytest.approx(base["probabilistic_sharpe"], abs=1e-9)
    assert many["deflated_sharpe"] < base["deflated_sharpe"]


def test_deflated_sharpe_monotonic_in_trials() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, size=250))
    d1 = compute_metrics(returns, _equity(returns), n_trials=1)["deflated_sharpe"]
    d10 = compute_metrics(returns, _equity(returns), n_trials=10)["deflated_sharpe"]
    d100 = compute_metrics(returns, _equity(returns), n_trials=100)["deflated_sharpe"]
    assert d1 >= d10 >= d100


def test_zero_variance_returns_zero() -> None:
    returns = pd.Series([0.0] * 50)
    metrics = compute_metrics(returns, _equity(returns))
    assert metrics["probabilistic_sharpe"] == 0.0
    assert metrics["deflated_sharpe"] == 0.0


def test_empty_metrics_has_new_keys() -> None:
    metrics = compute_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert metrics["probabilistic_sharpe"] == 0.0
    assert metrics["deflated_sharpe"] == 0.0


def test_n_trials_below_one_raises() -> None:
    returns = pd.Series([0.01, -0.01] * 30)
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        compute_metrics(returns, _equity(returns), n_trials=0)


def test_n_trials_two_deflates_below_psr() -> None:
    # First call through the full benchmark formula (no short-circuit). With a
    # positive-mean series the deflated Sharpe must drop below the PSR.
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, size=250))
    metrics = compute_metrics(returns, _equity(returns), n_trials=2)
    assert metrics["deflated_sharpe"] < metrics["probabilistic_sharpe"]


# --- golden values: pin the formulas, not just their properties ------------


def test_sharpe_and_sortino_golden_values() -> None:
    # mean = 0.01; ddof=0 variance = 2.5e-4 -> sharpe = sqrt(252 * 0.4) = sqrt(100.8)
    # downside (clip upper 0) = [0, 0, -0.01, 0]; ddof=0 std = sqrt(1.875e-5)
    #   -> sortino = sqrt(252 * (0.01^2 / 1.875e-5)) = sqrt(1344)
    returns = pd.Series([0.02, 0.0, -0.01, 0.03])
    metrics = compute_metrics(returns, _equity(returns))
    assert metrics["sharpe"] == pytest.approx(np.sqrt(100.8), rel=1e-9)
    assert metrics["sortino"] == pytest.approx(np.sqrt(1344.0), rel=1e-9)
    assert metrics["n_bars"] == 4


def test_annualized_return_compounds_geometrically() -> None:
    # growth = 1.02 * 1.00 * 0.99 * 1.03; annualized over 4 bars at 252/yr
    returns = pd.Series([0.02, 0.0, -0.01, 0.03])
    metrics = compute_metrics(returns, _equity(returns))
    expected = (1.02 * 1.00 * 0.99 * 1.03) ** (252 / 4) - 1.0
    assert metrics["annualized_return"] == pytest.approx(expected, rel=1e-9)


def test_hit_rate_counts_only_active_bars() -> None:
    # zero-return (flat) bars are excluded: 2 wins / 3 nonzero bars
    returns = pd.Series([0.01, 0.0, -0.02, 0.03])
    metrics = compute_metrics(returns, _equity(returns))
    assert metrics["hit_rate"] == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_max_drawdown_peak_to_trough() -> None:
    # peak 120 -> trough 80: max drawdown = 80/120 - 1 = -1/3
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0])
    returns = equity.pct_change().fillna(0.0)
    metrics = compute_metrics(returns, equity)
    assert metrics["max_drawdown"] == pytest.approx(-1.0 / 3.0, rel=1e-9)


def test_max_drawdown_counts_loss_from_first_bar() -> None:
    # a curve that never exceeds its starting value still reports its drawdown
    equity = pd.Series([90.0, 80.0, 95.0])
    returns = equity.pct_change().fillna(0.0)
    metrics = compute_metrics(returns, equity)
    assert metrics["max_drawdown"] == pytest.approx(80.0 / 90.0 - 1.0, rel=1e-9)
