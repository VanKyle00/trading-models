"""Tests for the bootstrap test, Benjamini-Hochberg FDR, and trade metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.validation import bootstrap_t_test


def test_bootstrap_ci_brackets_mean_and_pvalue_small_for_strong_signal() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.02, 0.01, size=500))
    t_stat, ci_lo, ci_hi, p_value = bootstrap_t_test(rets, n_boot=2000, seed=1)

    assert ci_lo < float(rets.mean()) < ci_hi
    assert ci_lo > 0.0  # CI excludes zero
    assert 0.0 < p_value <= 2.0 / 2001  # floored, strictly positive, tiny
    assert t_stat > 0.0


def test_bootstrap_pvalue_large_for_zero_mean() -> None:
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.0, 0.05, size=400))
    _, ci_lo, ci_hi, p_value = bootstrap_t_test(rets, n_boot=2000, seed=3)
    assert ci_lo < 0.0 < ci_hi
    assert 0.0 < p_value <= 1.0
    assert p_value > 0.05


def test_bootstrap_handles_tiny_sample() -> None:
    rets = pd.Series([0.01])
    t_stat, ci_lo, ci_hi, p_value = bootstrap_t_test(rets, n_boot=100, seed=0)
    assert t_stat == 0.0 and ci_lo == 0.0 and ci_hi == 0.0 and p_value == 1.0
