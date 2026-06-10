"""Tests for the fundamental-analysis gate (Stage 1 of the swing scanner)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.scanner.fa_gate import apply_edgar_trends, score_fundamentals


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


def _trends(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": t, **vals} for t, vals in rows.items()],
        columns=["ticker", "revenue_yoy", "revenue_yoy_prev", "revenue_accel", "eps_change_yoy"],
    )


def test_edgar_trends_rerank_the_gate() -> None:
    # YF metrics put A barely ahead of B; B's accelerating revenue and EPS
    # momentum from EDGAR must overtake A in the blended score.
    uni = _universe(["A", "B"], ["Tech", "Tech"])
    fnd = _fundamentals({"A": {"revenue_growth": 0.12}, "B": {"revenue_growth": 0.11}})
    scored = score_fundamentals(uni, fnd, keep=2)
    trends = _trends(
        {
            "A": {"revenue_yoy": 0.02, "revenue_accel": -0.05, "eps_change_yoy": -0.2},
            "B": {"revenue_yoy": 0.30, "revenue_accel": 0.10, "eps_change_yoy": 0.5},
        }
    )

    out = apply_edgar_trends(scored, trends, keep=1).set_index("ticker")

    assert out.loc["B", "fa_score"] > out.loc["A", "fa_score"]
    assert bool(out.loc["B", "passed_gate"])
    assert not bool(out.loc["A", "passed_gate"])


def test_edgar_trends_missing_ticker_keeps_yf_score() -> None:
    uni = _universe(["A", "B"], ["Tech", "Tech"])
    fnd = _fundamentals({"A": {}, "B": {}})
    scored = score_fundamentals(uni, fnd, keep=2)
    before = scored.set_index("ticker").loc["A", "fa_score"]

    out = apply_edgar_trends(scored, _trends({"B": {"revenue_yoy": 0.2}}), keep=2)
    out = out.set_index("ticker")

    assert out.loc["A", "fa_score"] == before  # no EDGAR data -> unblended
    assert bool(out.loc["A", "passed_gate"])  # still ranked, not dropped


def test_edgar_trends_only_reranks_pass_one_survivors() -> None:
    uni = _universe(["A", "B", "C"], ["Tech"] * 3)
    fnd = _fundamentals(
        {
            "A": {"revenue_growth": 0.30},
            "B": {"revenue_growth": 0.20},
            "C": {"total_revenue": 0.0},  # hard-filtered in pass 1
        }
    )
    scored = score_fundamentals(uni, fnd, keep=3)

    out = apply_edgar_trends(
        scored, _trends({"C": {"revenue_yoy": 9.9, "revenue_accel": 9.9}}), keep=2
    ).set_index("ticker")

    assert not bool(out.loc["C", "passed_gate"])  # great EDGAR data can't resurrect it
    assert out["passed_gate"].sum() == 2


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


def test_short_gate_keeps_worst_fundamentals() -> None:
    tickers = [f"T{i}" for i in range(6)]
    uni = _universe(tickers, ["Tech"] * 6)
    fnd = _fundamentals({t: {"revenue_growth": 0.05 + i * 0.01} for i, t in enumerate(tickers)})

    out = score_fundamentals(uni, fnd, keep=2, short_keep=2).set_index("ticker")

    assert bool(out.loc["T0", "passed_short_gate"])  # worst growth
    assert bool(out.loc["T1", "passed_short_gate"])
    assert out["passed_short_gate"].sum() == 2
    assert not (out["passed_gate"] & out["passed_short_gate"]).any()  # disjoint cohorts


def test_stance_column_labels_both_cohorts() -> None:
    tickers = [f"T{i}" for i in range(4)]
    uni = _universe(tickers, ["Tech"] * 4)
    fnd = _fundamentals({t: {"revenue_growth": 0.05 + i * 0.01} for i, t in enumerate(tickers)})

    out = score_fundamentals(uni, fnd, keep=1, short_keep=1).set_index("ticker")

    assert out.loc["T3", "stance"] == "long"
    assert out.loc["T0", "stance"] == "short"
    assert out["stance"].isna().sum() == 2


def test_short_gate_excludes_sparse_coverage() -> None:
    # bad data must not masquerade as bad fundamentals: a ticker failing the
    # coverage filter is ineligible for the short side too
    uni = _universe(["FULL", "SPARSE"], ["Tech", "Tech"])
    fnd = _fundamentals(
        {
            "FULL": {},
            "SPARSE": {
                "revenue_growth": np.nan,
                "earnings_growth": np.nan,
                "operating_margin": np.nan,
            },
        }
    )

    out = score_fundamentals(uni, fnd, keep=1, short_keep=2).set_index("ticker")

    assert not bool(out.loc["SPARSE", "passed_short_gate"])
    assert out["passed_short_gate"].sum() == 0  # FULL is long, SPARSE ineligible


def test_short_gate_off_by_default() -> None:
    uni = _universe(["A", "B"], ["Tech", "Tech"])
    fnd = _fundamentals({"A": {}, "B": {}})

    out = score_fundamentals(uni, fnd, keep=1)

    assert "passed_short_gate" in out.columns
    assert out["passed_short_gate"].sum() == 0
    assert (out["stance"].dropna() == "long").all()


def test_edgar_trends_decelerating_revenue_improves_short_rank() -> None:
    # S1 and S2 are both short candidates; S1's trends are improving (bad
    # short), S2's are deteriorating (good short). EDGAR must keep S2.
    uni = _universe(["LONGY", "S1", "S2"], ["Tech"] * 3)
    fnd = _fundamentals(
        {
            "LONGY": {"revenue_growth": 0.30},
            "S1": {"revenue_growth": 0.02},
            "S2": {"revenue_growth": 0.03},
        }
    )
    scored = score_fundamentals(uni, fnd, keep=1, short_keep=2)
    trends = _trends(
        {
            "S1": {"revenue_yoy": 0.30, "revenue_accel": 0.10, "eps_change_yoy": 0.5},
            "S2": {"revenue_yoy": -0.20, "revenue_accel": -0.10, "eps_change_yoy": -0.4},
        }
    )

    out = apply_edgar_trends(scored, trends, keep=1, short_keep=1).set_index("ticker")

    assert bool(out.loc["S2", "passed_short_gate"])
    assert not bool(out.loc["S1", "passed_short_gate"])
    assert out.loc["S2", "stance"] == "short"


def test_edgar_trends_missing_short_data_keeps_unblended_score() -> None:
    uni = _universe(["LONGY", "S1", "S2"], ["Tech"] * 3)
    fnd = _fundamentals(
        {
            "LONGY": {"revenue_growth": 0.30},
            "S1": {"revenue_growth": 0.02},
            "S2": {"revenue_growth": 0.03},
        }
    )
    scored = score_fundamentals(uni, fnd, keep=1, short_keep=2)
    before = scored.set_index("ticker").loc["S1", "fa_score"]

    out = apply_edgar_trends(scored, _trends({}), keep=1, short_keep=2).set_index("ticker")

    assert out.loc["S1", "fa_score"] == before  # no EDGAR data -> unblended
    assert bool(out.loc["S1", "passed_short_gate"])  # still ranked, not dropped


def test_edgar_trends_cannot_resurrect_failed_short_candidates() -> None:
    uni = _universe(["LONGY", "S1", "INELIGIBLE"], ["Tech"] * 3)
    fnd = _fundamentals(
        {
            "LONGY": {"revenue_growth": 0.30},
            "S1": {"revenue_growth": 0.02},
            "INELIGIBLE": {"total_revenue": 0.0},  # hard-filtered in pass 1
        }
    )
    scored = score_fundamentals(uni, fnd, keep=1, short_keep=2)

    out = apply_edgar_trends(
        scored,
        _trends({"INELIGIBLE": {"revenue_yoy": -9.9, "revenue_accel": -9.9}}),
        keep=1,
        short_keep=2,
    ).set_index("ticker")

    assert not bool(out.loc["INELIGIBLE", "passed_short_gate"])


def test_two_sided_gate_disjoint_under_overflow() -> None:
    # regression for the ~passed_gate guard: 3 eligible, keep=2, short_keep=2
    # must yield 2 longs + 1 short with zero overlap (longs take precedence
    # on ties because method="first" ranks do not mirror)
    uni = _universe(["A", "B", "C"], ["Tech"] * 3)
    fnd = _fundamentals({"A": {}, "B": {}, "C": {}})

    out = score_fundamentals(uni, fnd, keep=2, short_keep=2)

    assert out["passed_gate"].sum() == 2
    assert out["passed_short_gate"].sum() == 1
    assert not (out["passed_gate"] & out["passed_short_gate"]).any()


def test_edgar_no_data_short_outranks_improving_short() -> None:
    # an improving-trend short loses its slot to a no-data short: the blend
    # raises the improving name's score and ascending rank prefers the lower
    uni = _universe(["LONGY", "S1", "S2"], ["Tech"] * 3)
    fnd = _fundamentals(
        {
            "LONGY": {"revenue_growth": 0.30},
            "S1": {"revenue_growth": 0.02},
            "S2": {"revenue_growth": 0.03},
        }
    )
    scored = score_fundamentals(uni, fnd, keep=1, short_keep=2)
    trends = _trends(
        {"S2": {"revenue_yoy": 0.30, "revenue_accel": 0.10, "eps_change_yoy": 0.5}}
    )

    out = apply_edgar_trends(scored, trends, keep=1, short_keep=1).set_index("ticker")

    assert bool(out.loc["S1", "passed_short_gate"])  # no-data short keeps the slot
    assert not bool(out.loc["S2", "passed_short_gate"])
