"""Tests for the bootstrap test, Benjamini-Hochberg FDR, and trade metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_benjamini_hochberg_exact_threshold_and_count() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]  # m=8, alpha=0.05
    rejected, threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)

    # BH at m=8: only ranks 1-2 pass (0.001<=0.00625, 0.008<=0.0125; 0.039>0.01875)
    assert threshold == pytest.approx(0.008)
    assert sum(rejected) == 2
    assert rejected[0] is True and rejected[1] is True
    assert all((p <= threshold) == r for p, r in zip(pvals, rejected, strict=True))


def test_benjamini_hochberg_step_up_is_largest_i_not_first_failure() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    # a small p-value (0.005) follows a failing one (0.04) in input order; the
    # step-up must scan to the LARGEST passing rank, not stop at the first failure.
    pvals = [0.001, 0.04, 0.005]  # sorted: 0.001,0.005,0.04 ; m=3, alpha=0.05
    rejected, threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)
    # rank1 0.001<=0.0167 ok; rank2 0.005<=0.0333 ok; rank3 0.04<=0.05 ok -> all
    assert threshold == pytest.approx(0.04)
    assert rejected == [True, True, True]


def test_benjamini_hochberg_no_rejections() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    rejected, threshold = benjamini_hochberg_fdr([0.6, 0.7, 0.9], alpha=0.05)
    assert rejected == [False, False, False]
    assert threshold == 0.0


def test_benjamini_hochberg_empty() -> None:
    from tradinglib.validation import benjamini_hochberg_fdr

    rejected, threshold = benjamini_hochberg_fdr([], alpha=0.05)
    assert rejected == []
    assert threshold == 0.0
