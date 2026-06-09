"""Tests for the fundamental-analysis gate (Stage 1 of the swing scanner)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.scanner.fa_gate import score_fundamentals


def _universe(tickers: list[str], sectors: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": tickers,
            "name": tickers,
            "sector": sectors,
            "sub_industry": sectors,
            "cik": range(1, len(tickers) + 1),
        }
    )


def _fundamentals(rows: dict[str, dict]) -> pd.DataFrame:
    defaults = {
        "snapshot": "2026-06-09",
        "market_cap": 1e11,
        "total_revenue": 1e10,
        "revenue_growth": 0.10,
        "earnings_growth": 0.10,
        "operating_margin": 0.20,
        "debt_to_equity": 100.0,
        "free_cashflow": 5e9,
        "forward_pe": 20.0,
        "trailing_pe": 25.0,
        "roe": 0.25,
        "avg_volume": 1e7,
    }
    out = []
    for ticker, overrides in rows.items():
        row = {"ticker": ticker, **defaults, **overrides}
        out.append(row)
    return pd.DataFrame(out)


def test_output_columns_and_gate_size() -> None:
    tickers = [f"T{i}" for i in range(6)]
    uni = _universe(tickers, ["Tech"] * 6)
    fnd = _fundamentals({t: {"revenue_growth": 0.05 + i * 0.01} for i, t in enumerate(tickers)})

    out = score_fundamentals(uni, fnd, keep=3)

    for col in ("fa_score", "fa_rank", "passed_gate", "fcf_yield"):
        assert col in out.columns
    assert out["passed_gate"].sum() == 3


def test_better_metrics_score_higher() -> None:
    uni = _universe(["GOOD", "BAD"], ["Tech", "Tech"])
    fnd = _fundamentals(
        {
            "GOOD": {"revenue_growth": 0.30, "debt_to_equity": 20.0, "forward_pe": 15.0},
            "BAD": {"revenue_growth": 0.01, "debt_to_equity": 300.0, "forward_pe": 40.0},
        }
    )

    out = score_fundamentals(uni, fnd, keep=1).set_index("ticker")

    assert out.loc["GOOD", "fa_score"] > out.loc["BAD", "fa_score"]
    assert bool(out.loc["GOOD", "passed_gate"])
    assert not bool(out.loc["BAD", "passed_gate"])


def test_forward_pe_is_sector_relative() -> None:
    # SOFT trades at 30x in a sector where peers trade at 35-40x (cheap for
    # its sector); MINE trades at 12x where peers trade at 8-10x (expensive
    # for its sector). Sector-relative scoring must reward SOFT.
    uni = _universe(
        ["SOFT", "S2", "S3", "MINE", "M2", "M3"],
        ["Tech", "Tech", "Tech", "Materials", "Materials", "Materials"],
    )
    fnd = _fundamentals(
        {
            "SOFT": {"forward_pe": 30.0},
            "S2": {"forward_pe": 35.0},
            "S3": {"forward_pe": 40.0},
            "MINE": {"forward_pe": 12.0},
            "M2": {"forward_pe": 8.0},
            "M3": {"forward_pe": 10.0},
        }
    )

    out = score_fundamentals(uni, fnd, keep=6).set_index("ticker")

    assert out.loc["SOFT", "pct_forward_pe"] > out.loc["MINE", "pct_forward_pe"]


def test_sparse_coverage_fails_gate() -> None:
    uni = _universe(["FULL", "SPARSE"], ["Tech", "Tech"])
    fnd = _fundamentals(
        {
            "FULL": {},
            "SPARSE": {
                "revenue_growth": np.nan,
                "earnings_growth": np.nan,
                "operating_margin": np.nan,
                # debt, fcf, pe present -> only 3 of 6 metrics
            },
        }
    )

    out = score_fundamentals(uni, fnd, keep=10).set_index("ticker")

    assert bool(out.loc["FULL", "passed_gate"])
    assert not bool(out.loc["SPARSE", "passed_gate"])


def test_nonpositive_revenue_fails_gate() -> None:
    uni = _universe(["OK", "NOREV"], ["Tech", "Tech"])
    fnd = _fundamentals({"OK": {}, "NOREV": {"total_revenue": 0.0}})

    out = score_fundamentals(uni, fnd, keep=10).set_index("ticker")

    assert bool(out.loc["OK", "passed_gate"])
    assert not bool(out.loc["NOREV", "passed_gate"])


def test_missing_metric_does_not_zero_the_score() -> None:
    # A ticker missing one metric is scored on the mean of what it has,
    # not penalized to zero.
    uni = _universe(["A", "B", "C"], ["Tech"] * 3)
    fnd = _fundamentals(
        {
            "A": {"revenue_growth": 0.30, "free_cashflow": np.nan, "earnings_growth": np.nan},
            "B": {"revenue_growth": 0.20},
            "C": {"revenue_growth": 0.10},
        }
    )

    out = score_fundamentals(uni, fnd, keep=3).set_index("ticker")

    assert not np.isnan(out.loc["A", "fa_score"])
    assert out.loc["A", "fa_score"] > 0
