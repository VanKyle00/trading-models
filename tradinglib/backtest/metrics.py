"""Backtest performance metrics — kept identical across every model so results
are comparable. See ``docs/methodology.md`` for assumptions and conventions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe benchmark.
_EULER_MASCHERONI = 0.5772156649015329


def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
    n_trials: int = 1,
) -> dict:
    """Standard performance metrics on a per-bar return series.

    Returns a plain dict (JSON-serializable) so each model can dump it to
    ``results/metrics.json`` without further conversion.

    ``n_trials`` is the number of independent strategy configurations tried to
    arrive at this one; it deflates the Sharpe via the Deflated Sharpe Ratio
    (Bailey & López de Prado, 2014). ``n_trials=1`` reduces the Deflated Sharpe
    to the Probabilistic Sharpe Ratio (benchmark 0).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if len(returns) == 0:
        return _empty_metrics()

    mean = float(returns.mean())
    std = float(returns.std(ddof=0))

    sharpe = float(np.sqrt(periods_per_year) * mean / std) if std > 0 else 0.0

    downside_std = float(returns.clip(upper=0.0).std(ddof=0))
    sortino = float(np.sqrt(periods_per_year) * mean / downside_std) if downside_std > 0 else 0.0

    # Annualized return from compounding the realized returns
    total_growth = float((1.0 + returns).prod())
    ann_return = (
        float(total_growth ** (periods_per_year / len(returns)) - 1.0) if total_growth > 0 else -1.0
    )

    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    max_drawdown = float(drawdowns.min())

    # Hit rate measured only over bars where the strategy was actually active
    nonzero = returns[returns != 0.0]
    hit_rate = float((nonzero > 0).mean()) if len(nonzero) > 0 else 0.0

    psr, dsr = _probabilistic_and_deflated_sharpe(returns, mean, std, n_trials)

    return {
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "probabilistic_sharpe": psr,
        "deflated_sharpe": dsr,
        "n_bars": len(returns),
    }


def _probabilistic_and_deflated_sharpe(
    returns: pd.Series, mean: float, std: float, n_trials: int
) -> tuple[float, float]:
    """Return (probabilistic_sharpe, deflated_sharpe).

    Both use the *non-annualized* per-bar Sharpe and correct for the skew and
    kurtosis of the return series (Bailey & López de Prado, 2014). The
    Probabilistic Sharpe is P(true Sharpe > 0). The Deflated Sharpe raises the
    benchmark to the expected maximum Sharpe across ``n_trials`` attempts.
    """
    n = len(returns)
    if std <= 0.0 or n < 2:
        return 0.0, 0.0

    sr = mean / std  # non-annualized per-bar Sharpe

    skew = float(returns.skew()) if n > 2 else 0.0
    excess_kurt = float(returns.kurt()) if n > 3 else 0.0
    if math.isnan(skew):
        skew = 0.0
    if math.isnan(excess_kurt):
        excess_kurt = 0.0
    kurt = excess_kurt + 3.0  # pandas .kurt() is excess; the formula needs Pearson (normal=3)

    # Variance of the Sharpe estimator (Lo 2002 / Bailey-LdP), per bar.
    sr_var = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) / (n - 1)
    # The Lo approximation's variance can go negative for extreme skew/kurtosis
    # combinations (outside its valid domain). Return 0 as a conservative
    # sentinel rather than a spurious probability.
    if sr_var <= 0.0:
        return 0.0, 0.0
    sr_sigma = math.sqrt(sr_var)

    psr = float(norm.cdf(sr / sr_sigma))

    if n_trials <= 1:
        return psr, psr

    # Expected maximum of n_trials i.i.d. Sharpe estimates ~ N(0, sr_var).
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    sr_benchmark = sr_sigma * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)
    dsr = float(norm.cdf((sr - sr_benchmark) / sr_sigma))
    return psr, dsr


def bootstrap_t_test(
    returns: pd.Series,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """Non-parametric bootstrap of the mean per-trade return.

    Returns ``(t_stat, ci_lower, ci_upper, p_value)``.

    - ``p_value``: a *centered* bootstrap test of H0:mean=0 — resample the
      mean-centered returns and count how often ``|boot_mean*| >= |observed_mean|``,
      smoothed by ``1/(n_boot+1)`` so it is never exactly 0 (a bootstrap p-value
      floor; report as ``< 1/n_boot`` when it hits the floor).
    - ``(ci_lower, ci_upper)``: a percentile CI from the *uncentered* resampled
      means — an interval estimate computed by a different (internally consistent)
      procedure than the p-value; the two are not guaranteed to agree sign-for-sign
      in finite samples.
    - ``t_stat``: classic Student t (mean / (std/sqrt(n))), reported as a
      descriptive statistic only — it is NOT the test statistic (per-trade returns
      are fat-tailed and few, Ch. 4).

    Tiny samples (n < 2) return the conservative sentinel ``(0.0, 0.0, 0.0, 1.0)``.
    """
    x = returns.to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0

    mean = float(x.mean())
    std = float(x.std(ddof=1))
    t_stat = float(mean / (std / math.sqrt(n))) if std > 0 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)

    alpha = 1.0 - confidence
    ci_lower = float(np.quantile(boot_means, alpha / 2.0))
    ci_upper = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    # centered bootstrap p-value (H0: mean == 0)
    shifted = boot_means - mean
    extreme = int((np.abs(shifted) >= abs(mean)).sum())
    p_value = (extreme + 1) / (n_boot + 1)
    return t_stat, ci_lower, ci_upper, float(p_value)


def benjamini_hochberg_fdr(pvalues: list[float], alpha: float = 0.05) -> tuple[list[bool], float]:
    """Benjamini-Hochberg FDR control over a set of hypotheses.

    Returns ``(rejected, threshold)`` where ``rejected[i]`` corresponds to
    ``pvalues[i]`` (input order preserved) and ``threshold`` is the largest
    p-value passing the BH step-up (0.0 if none). Scans to the LARGEST rank i with
    ``p(i) <= (i/m)*alpha`` and rejects all p <= that threshold. Controls the
    expected false-discovery rate at ``alpha`` across the hypothesis set (Ch. 4).
    """
    m = len(pvalues)
    if m == 0:
        return [], 0.0

    order = sorted(range(m), key=lambda i: pvalues[i])
    threshold = 0.0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= (rank / m) * alpha:
            threshold = pvalues[i]
    rejected = [p <= threshold and threshold > 0.0 for p in pvalues]
    return rejected, float(threshold)


def _empty_metrics() -> dict:
    return {
        "annualized_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "probabilistic_sharpe": 0.0,
        "deflated_sharpe": 0.0,
        "n_bars": 0,
    }
