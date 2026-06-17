"""B-series parity guarantee (audit #88/#89/#90).

The B-series is ADDITIVE: it adds a leak/honesty flag (metadata), an as-of EDGAR
path (off by default), and unadjusted price columns — without changing any
pre-existing numeric output. The bulk of that guarantee is the rest of the test
suite: every pre-B assertion that pins an R / level / DSR / YoY / loader value
still passes unchanged. This file consolidates the two cross-cutting invariants
that span the new code paths.

The one INTENDED numeric change — the ledger scoring corp-action tickets against
unadjusted bars instead of dropping them — is asserted separately in
test_scanner_ledger.py::test_ledger_scores_against_unadjusted_when_present.
"""

from __future__ import annotations

import pandas as pd


def _frame_facts() -> dict:
    def usd(frame: str, end: str, val: float) -> dict:
        return {"end": end, "val": val, "form": "10-Q", "frame": frame}

    return {
        "cik": 320193,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            usd("CY2025Q1", "2025-03-29", 120e9),
                            usd("CY2026Q1", "2026-03-28", 150e9),
                        ]
                    }
                }
            }
        },
    }


def test_edgar_asof_none_is_the_unchanged_frame_path() -> None:
    # asof=None routes the SEC-frame path exactly as before — the new units-based
    # as-of code never runs, so live scanning is byte-identical.
    from tradinglib.loaders.fundamentals.edgar import _trends_from_facts

    facts = _frame_facts()
    assert _trends_from_facts(facts) == _trends_from_facts(facts, asof=None)
    assert _trends_from_facts(facts)["revenue_yoy"] == 150 / 120 - 1


def test_unadjusted_attachment_leaves_adjusted_byte_identical() -> None:
    # Adding unadj_* must not perturb a single adjusted value — indicators, signals,
    # and the tournament all read the adjusted columns and must stay identical.
    from tradinglib.loaders.equities.yfinance import _attach_unadjusted, _canonicalize

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    adj = pd.DataFrame(
        {
            ("Open", "SPY"): [50.0, 51.0, 52.0],
            ("High", "SPY"): [50.5, 51.5, 52.5],
            ("Low", "SPY"): [49.5, 50.5, 51.5],
            ("Close", "SPY"): [50.0, 51.0, 52.0],
            ("Volume", "SPY"): [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )
    unadj = pd.DataFrame(
        {
            ("Open", "SPY"): [100.0, 102.0, 104.0],
            ("High", "SPY"): [101.0, 103.0, 105.0],
            ("Low", "SPY"): [99.0, 101.0, 103.0],
            ("Close", "SPY"): [100.0, 102.0, 104.0],
            ("Adj Close", "SPY"): [50.0, 51.0, 52.0],
            ("Volume", "SPY"): [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )
    base = _canonicalize(adj, "SPY")
    out = _attach_unadjusted(base.copy(), unadj)

    for col in ("open", "high", "low", "close", "volume", "symbol"):
        assert out[col].tolist() == base[col].tolist()
    assert {"unadj_open", "unadj_high", "unadj_low", "unadj_close"}.issubset(out.columns)
    assert out["unadj_close"].tolist() == [100.0, 102.0, 104.0]
