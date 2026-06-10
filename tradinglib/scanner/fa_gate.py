"""Stage 1: fundamental gate — cut the universe to a quant-scored shortlist.

Six metrics are scored as cross-sectional percentiles (forward P/E within
GICS sector, the rest across the full universe); ``fa_score`` is the mean of
the percentiles a ticker actually has. Hard filters: at least 4 of 6 metrics
present and positive trailing-twelve-month revenue. The top ``keep`` names by
``fa_score`` pass the gate.
"""

from __future__ import annotations

import pandas as pd

# metric column -> True if higher is better
_METRIC_DIRECTION: dict[str, bool] = {
    "revenue_growth": True,
    "earnings_growth": True,
    "operating_margin": True,
    "debt_to_equity": False,
    "fcf_yield": True,
    "forward_pe": False,
}
_MIN_COVERAGE = 4


def score_fundamentals(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    keep: int = 40,
    short_keep: int = 0,
) -> pd.DataFrame:
    """Merge universe + fundamentals and score every ticker.

    Returns one row per universe ticker with ``pct_<metric>`` percentile
    columns, ``fa_score``, ``fa_rank`` (1 = best) and ``passed_gate``. With
    ``short_keep > 0`` the gate is two-sided: the bottom ``short_keep``
    eligible names gain ``passed_short_gate`` / ``short_rank`` (1 = worst
    fundamentals), and ``stance`` labels each kept row ``"long"``/``"short"``.
    Eligibility (metric coverage, positive revenue) is identical on both
    sides — bad data must not masquerade as bad fundamentals.
    """
    df = universe.merge(fundamentals, on="ticker", how="left")
    df["fcf_yield"] = df["free_cashflow"] / df["market_cap"]

    for metric, higher_is_better in _METRIC_DIRECTION.items():
        series = df[metric]
        if metric == "forward_pe":
            pct = series.groupby(df["sector"]).rank(pct=True)
        else:
            pct = series.rank(pct=True)
        df[f"pct_{metric}"] = pct if higher_is_better else 1.0 - pct

    pct_cols = [f"pct_{m}" for m in _METRIC_DIRECTION]
    df["fa_score"] = df[pct_cols].mean(axis=1)

    coverage = df[list(_METRIC_DIRECTION)].notna().sum(axis=1)
    eligible = (coverage >= _MIN_COVERAGE) & (df["total_revenue"] > 0)

    df["fa_rank"] = df["fa_score"].where(eligible).rank(ascending=False, method="first")
    df["passed_gate"] = eligible & (df["fa_rank"] <= keep)
    df["short_rank"] = (
        df["fa_score"].where(eligible & ~df["passed_gate"]).rank(ascending=True, method="first")
    )
    df["passed_short_gate"] = eligible & ~df["passed_gate"] & (df["short_rank"] <= short_keep)
    df["stance"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[df["passed_gate"], "stance"] = "long"
    df.loc[df["passed_short_gate"], "stance"] = "short"
    return df.sort_values("fa_rank").reset_index(drop=True)


# EDGAR companyfacts trend metrics blended into the gate (all higher-is-better)
_EDGAR_METRICS = ("revenue_yoy", "revenue_accel", "eps_change_yoy")
_EDGAR_WEIGHT = 0.3


def _cohort_edgar_score(df: pd.DataFrame, mask: pd.Series, suffix: str) -> pd.Series:
    """Mean EDGAR-trend percentile within one cohort (higher = better trends)."""
    cols = []
    for metric in _EDGAR_METRICS:
        col = f"pct_{metric}{suffix}"
        df[col] = df[metric].where(mask).rank(pct=True)
        cols.append(col)
    return df[cols].mean(axis=1)


def apply_edgar_trends(
    scored: pd.DataFrame, trends: pd.DataFrame, *, keep: int, short_keep: int = 0
) -> pd.DataFrame:
    """Second gate pass: blend EDGAR quarterly trends and re-rank both cohorts.

    Trend percentiles are computed within each cohort and blended as
    ``0.7*fa_score + 0.3*edgar_score``. Longs re-rank descending (improving
    trends help); shorts re-rank ascending, so *decelerating* trends lower
    the blended score and improve the short rank — the sign flip falls out
    of the ranking direction, no negation needed. Tickers without EDGAR data
    keep their unblended score; pass-1 failures stay failed.
    """
    df = scored.merge(trends, on="ticker", how="left")
    passed = df["passed_gate"]
    passed_short = df["passed_short_gate"]

    df["edgar_score"] = _cohort_edgar_score(df, passed, "")
    df["edgar_score_short"] = _cohort_edgar_score(df, passed_short, "_short")

    blended_long = (1.0 - _EDGAR_WEIGHT) * df["fa_score"] + _EDGAR_WEIGHT * df["edgar_score"]
    blended_short = (1.0 - _EDGAR_WEIGHT) * df["fa_score"] + _EDGAR_WEIGHT * df["edgar_score_short"]
    df["fa_score_yf"] = df["fa_score"]
    score = df["fa_score"].copy()
    score = blended_long.where(passed & df["edgar_score"].notna(), score)
    score = blended_short.where(passed_short & df["edgar_score_short"].notna(), score)
    df["fa_score"] = score

    df["fa_rank"] = df["fa_score"].where(passed).rank(ascending=False, method="first")
    df["passed_gate"] = passed & (df["fa_rank"] <= keep)
    df["short_rank"] = df["fa_score"].where(passed_short).rank(ascending=True, method="first")
    df["passed_short_gate"] = passed_short & (df["short_rank"] <= short_keep)
    df["stance"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[df["passed_gate"], "stance"] = "long"
    df.loc[df["passed_short_gate"], "stance"] = "short"
    return df.sort_values("fa_rank").reset_index(drop=True)
