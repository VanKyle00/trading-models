"""Tests for the backfill replay's pooled-certified promotion (--pooled A/B arm)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_scan.py"


@pytest.fixture
def backfill():
    spec = importlib.util.spec_from_file_location("backfill_scan", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_scan"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["backfill_scan"]


def _watch_row(ticker: str, strategy: str, *, levels: dict | None) -> dict:
    """A setup watch row as build_watchlist emits it (the promotion input shape)."""
    return {
        "ticker": ticker,
        "stance": "long",
        "strategy": strategy,
        "tier": "watch",
        "tier_reason": "sub-threshold evidence",
        "entry_window": 7,
        "deflated_sharpe": 0.7,
        "levels": levels,
    }


def _certification(*, pead_certified: bool) -> dict:
    """A certification sidecar certifying (or not) pead:long; ma_pullback:long never certified."""
    return {
        "built_asof": "2026-01-30",
        "verdicts": {
            "pead:long": {
                "certified": pead_certified,
                "pooled_dsr": 0.96,
                "n_dates": 24,
                "total_r": 6.1,
                "reasons": [],
            },
            "ma_pullback:long": {
                "certified": False,
                "pooled_dsr": 0.5,
                "n_dates": 8,
                "total_r": 1.0,
                "reasons": ["n_dates 8 < 20"],
            },
        },
    }


def test_promote_certified_promotes_certified_pead(backfill) -> None:
    levels = {"entry": 120.0, "entry_type": "stop", "stop": 110.0, "target": 140.0}
    watchlist = {
        "long": [_watch_row("PEAD", "setup:pead", levels=levels)],
        "short": [],
    }
    input_long = watchlist["long"]
    input_len = len(input_long)
    kept, promoted = backfill._promote_certified(
        watchlist, _certification(pead_certified=True), "2026-01-30"
    )

    assert kept["long"] == []  # promoted row left the watchlist
    assert len(promoted) == 1
    row = promoted[0]
    assert set(row) == {
        "date",
        "ticker",
        "stance",
        "strategy",
        "tier",
        "tier_reason",
        "entry_window",
        "levels",
    }
    assert row["ticker"] == "PEAD"
    assert row["tier"] == "ticket"
    assert row["tier_reason"] == "pooled-certified"
    assert row["entry_window"] == 7  # preserved from the watch row
    assert row["strategy"] == "setup:pead"
    assert row["levels"] == levels
    assert row["date"] == "2026-01-30"
    # non-mutation: the original list object and its length are unchanged
    assert watchlist["long"] is input_long
    assert len(input_long) == input_len


def test_promote_certified_keeps_uncertified(backfill) -> None:
    levels = {"entry": 50.0, "entry_type": "stop", "stop": 45.0, "target": 60.0}
    watchlist = {
        "long": [_watch_row("MAPB", "setup:ma_pullback", levels=levels)],
        "short": [],
    }
    kept, promoted = backfill._promote_certified(
        watchlist, _certification(pead_certified=True), "2026-01-30"
    )

    assert promoted == []
    assert len(kept["long"]) == 1  # uncertified type stays on the watchlist


def test_promote_certified_keeps_levels_less_rows(backfill) -> None:
    """A certified type with no actionable levels is report-only — never promoted."""
    watchlist = {
        "long": [_watch_row("PEAD", "setup:pead", levels=None)],
        "short": [],
    }
    kept, promoted = backfill._promote_certified(
        watchlist, _certification(pead_certified=True), "2026-01-30"
    )

    assert promoted == []
    assert len(kept["long"]) == 1


def test_promote_certified_ignores_non_setup_rows(backfill) -> None:
    """Tournament-derived watch rows (strategy without a setup: prefix) never promote."""
    levels = {"entry": 120.0, "entry_type": "market", "stop": 110.0, "target": 140.0}
    watchlist = {
        "long": [_watch_row("AAPL", "sma_cross", levels=levels)],
        "short": [],
    }
    kept, promoted = backfill._promote_certified(
        watchlist, _certification(pead_certified=True), "2026-01-30"
    )

    assert promoted == []
    assert len(kept["long"]) == 1
