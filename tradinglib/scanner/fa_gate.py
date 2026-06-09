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
    universe: pd.DataFrame, fundamentals: pd.DataFrame, *, keep: int = 40
) -> pd.DataFrame:
    """Merge universe + fundamentals and score every ticker.

    Returns one row per universe ticker with ``pct_<metric>`` percentile
    columns, ``fa_score``, ``fa_rank`` (1 = best) and ``passed_gate``.
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
    return df.sort_values("fa_rank").reset_index(drop=True)


# EDGAR companyfacts trend metrics blended into the gate (all higher-is-better)
_EDGAR_METRICS = ("revenue_yoy", "revenue_accel", "eps_change_yoy")
_EDGAR_WEIGHT = 0.3


def apply_edgar_trends(scored: pd.DataFrame, trends: pd.DataFrame, *, keep: int) -> pd.DataFrame:
    """Second gate pass: blend EDGAR quarterly trends and re-rank the survivors.

    ``trends`` has one row per ticker (``revenue_yoy``, ``revenue_accel``,
    ``eps_change_yoy``); metrics are percentiled among pass-1 survivors and
    blended as ``0.7*fa_score + 0.3*edgar_score``. Tickers without EDGAR data
    keep their unblended score; tickers that failed pass 1 stay failed.
    """
    df = scored.merge(trends, on="ticker", how="left")
    passed = df["passed_gate"]

    pct_cols = []
    for metric in _EDGAR_METRICS:
        col = f"pct_{metric}"
        df[col] = df[metric].where(passed).rank(pct=True)
        pct_cols.append(col)
    df["edgar_score"] = df[pct_cols].mean(axis=1)

    blended = (1.0 - _EDGAR_WEIGHT) * df["fa_score"] + _EDGAR_WEIGHT * df["edgar_score"]
    df["fa_score_yf"] = df["fa_score"]
    df["fa_score"] = blended.where(passed & df["edgar_score"].notna(), df["fa_score"])

    df["fa_rank"] = df["fa_score"].where(passed).rank(ascending=False, method="first")
    df["passed_gate"] = passed & (df["fa_rank"] <= keep)
    return df.sort_values("fa_rank").reset_index(drop=True)
