"""Tests for nightly FDR tiering over tournament results."""

from __future__ import annotations

import pytest

from tradinglib.scanner.tiers import ENTRY_WINDOWS, apply_fdr, build_watchlist


def _entry(ticker, stance, dsrs, survivors=(), winner_levels=None):
    verdicts = [
        {"strategy": f"s{i}", "deflated_sharpe": d, "sharpe": d, "survived": f"s{i}" in survivors}
        for i, d in enumerate(dsrs)
    ]
    winner = None
    if survivors and winner_levels is not None:
        winner = {"strategy": survivors[0], "levels": winner_levels}
    return {
        "ticker": ticker,
        "stance": stance,
        "winner": winner,
        "survivors": list(survivors),
        "verdicts": verdicts,
    }


_LEVELS = {"entry": 100.0, "entry_type": "stop", "stop": 95.0, "target": 110.0}


def test_apply_fdr_passes_only_overwhelming_evidence() -> None:
    # one near-certain entry among many nulls: it must pass; weak ones must not
    tournament = {
        "long": [_entry("AAA", "long", [0.999])]
        + [_entry(f"N{i}", "long", [0.30]) for i in range(20)],
        "short": [],
    }
    passed, threshold, family = apply_fdr(tournament, alpha=0.10)
    assert family == 21
    assert passed[("long", "AAA")] is True
    assert not any(passed[("long", f"N{i}")] for i in range(20))
    assert 0.0 < threshold <= 0.10


def test_apply_fdr_skips_entries_without_verdicts() -> None:
    tournament = {"long": [{"ticker": "X", "stance": "long", "verdicts": []}], "short": []}
    passed, _, family = apply_fdr(tournament, alpha=0.10)
    assert family == 0 and passed == {}


def test_watchlist_reasons_and_levels() -> None:
    tournament = {
        "long": [
            _entry("FDRFAIL", "long", [0.95], survivors=("s0",), winner_levels=_LEVELS),
            _entry("NOLEVELS", "long", [0.96], survivors=("s0",), winner_levels=None),
            _entry("SETUPISH", "long", [0.60]),
            _entry("WEAK", "long", [0.20]),
        ],
        "short": [],
    }
    candidates = [
        {
            "ticker": "SETUPISH",
            "setups": [
                {"setup_type": "pead", "score": 0.7, "trigger_level": 50.0, "stop_level": 46.0}
            ],
        },
        {
            "ticker": "WEAK",
            "setups": [
                {
                    "setup_type": "ma_pullback",
                    "score": 0.5,
                    "trigger_level": 20.0,
                    "stop_level": 19.0,
                }
            ],
        },
    ]
    fdr_passed = {
        ("long", "FDRFAIL"): False,
        ("long", "NOLEVELS"): True,
        ("long", "SETUPISH"): False,
        ("long", "WEAK"): False,
    }
    wl = build_watchlist(tournament, candidates, fdr_passed, watch_dsr_floor=0.5)

    by_ticker = {w["ticker"]: w for w in wl["long"]}
    assert by_ticker["FDRFAIL"]["tier_reason"] == "failed nightly FDR"
    assert by_ticker["FDRFAIL"]["levels"] == _LEVELS  # ledger-trackable
    assert by_ticker["NOLEVELS"]["tier_reason"] == "no actionable entry tonight"
    assert by_ticker["NOLEVELS"]["levels"] is None  # report-only
    setup_row = by_ticker["SETUPISH"]
    assert setup_row["tier_reason"] == "sub-threshold evidence"
    assert setup_row["strategy"] == "setup:pead"
    assert setup_row["levels"] == {
        "entry": 50.0,
        "entry_type": "stop",
        "stop": 46.0,
        "target": 58.0,
    }  # 2R
    assert "WEAK" not in by_ticker  # best DSR 0.20 < floor
    assert all(w["tier"] == "watch" for w in wl["long"])
    assert wl["short"] == []


def test_watchlist_survivor_not_duplicated_by_setup_source() -> None:
    # a survivor demoted by FDR that ALSO fired a setup appears once (survivor source wins)
    tournament = {
        "long": [_entry("BOTH", "long", [0.95], survivors=("s0",), winner_levels=_LEVELS)],
        "short": [],
    }
    candidates = [
        {
            "ticker": "BOTH",
            "setups": [
                {"setup_type": "pead", "score": 0.7, "trigger_level": 50.0, "stop_level": 46.0}
            ],
        }
    ]
    wl = build_watchlist(tournament, candidates, {("long", "BOTH"): False}, watch_dsr_floor=0.5)
    assert len(wl["long"]) == 1
    assert wl["long"][0]["tier_reason"] == "failed nightly FDR"


def test_watchlist_tolerates_survivors_without_verdicts() -> None:
    # survivors list is non-empty but no matching verdict dict exists — must not raise
    tournament = {
        "long": [
            {
                "ticker": "MALFORMED",
                "stance": "long",
                "winner": None,
                "survivors": ["s0"],  # survivor key references a strategy not in verdicts
                "verdicts": [],  # empty — no verdict to match
            }
        ],
        "short": [],
    }
    fdr_passed: dict = {}
    wl = build_watchlist(tournament, [], fdr_passed, watch_dsr_floor=0.5)
    # must not raise and the malformed entry must be excluded (skipped, not crashed)
    assert "MALFORMED" not in {w["ticker"] for w in wl["long"]}


def test_watchlist_tolerates_candidate_without_setups() -> None:
    # a candidate with no setups (or empty list) must not raise
    tournament: dict = {"long": [], "short": []}
    candidates = [
        {"ticker": "NOSETUP", "setups": []},
        {"ticker": "ALSONONE"},  # missing key entirely
    ]
    fdr_passed: dict = {}
    wl = build_watchlist(tournament, candidates, fdr_passed, watch_dsr_floor=0.5)
    assert "NOSETUP" not in {w["ticker"] for w in wl["long"]}
    assert "ALSONONE" not in {w["ticker"] for w in wl["long"]}


def test_setup_watch_levels_short_target_below_trigger() -> None:
    from tradinglib.scanner.tiers import _setup_watch_levels

    setup = {"trigger_level": 50.0, "stop_level": 55.0}
    levels = _setup_watch_levels(setup, "short")
    assert levels["entry"] == 50.0
    assert levels["stop"] == 55.0
    assert levels["target"] == 40.0  # trigger - 2 * (stop - trigger)
    assert levels["entry_type"] == "stop"


def test_build_watchlist_sub_threshold_short_cohort() -> None:
    tournament = {
        "long": [],
        "short": [
            {
                "ticker": "WEAK",
                "stance": "short",
                "winner": None,
                "survivors": [],
                "verdicts": [{"strategy": "sma_cross", "deflated_sharpe": 0.7}],
            }
        ],
    }
    candidates = [
        {
            "ticker": "WEAK",
            "cohort": "short",
            "setups": [
                {
                    "setup_type": "base_breakdown",
                    "score": 0.8,
                    "trigger_level": 50.0,
                    "stop_level": 55.0,
                }
            ],
        }
    ]
    watchlist = build_watchlist(tournament, candidates, {}, watch_dsr_floor=0.5)
    assert len(watchlist["short"]) == 1
    row = watchlist["short"][0]
    assert row["strategy"] == "setup:base_breakdown"
    assert row["stance"] == "short"
    assert row["levels"]["target"] == 40.0


def test_setup_watch_levels_short_unattainable_target_is_none() -> None:
    from tradinglib.scanner.tiers import _setup_watch_levels

    # risk (1.6) > trigger/2 (1.5): the 2R target would be a negative price
    assert _setup_watch_levels({"trigger_level": 3.0, "stop_level": 4.6}, "short") is None


def test_build_watchlist_short_unattainable_target_is_report_only() -> None:
    tournament = {
        "long": [],
        "short": [
            {
                "ticker": "PENY",
                "stance": "short",
                "winner": None,
                "survivors": [],
                "verdicts": [{"strategy": "sma_cross", "deflated_sharpe": 0.7}],
            }
        ],
    }
    candidates = [
        {
            "ticker": "PENY",
            "cohort": "short",
            "setups": [
                {
                    "setup_type": "base_breakdown",
                    "score": 0.8,
                    "trigger_level": 3.0,
                    "stop_level": 4.6,
                }
            ],
        }
    ]
    watchlist = build_watchlist(tournament, candidates, {}, watch_dsr_floor=0.5)
    assert len(watchlist["short"]) == 1
    assert watchlist["short"][0]["levels"] is None  # report-only: no attainable 2R plan


def test_build_watchlist_rejects_unknown_cohort() -> None:
    candidates = [
        {
            "ticker": "TYPO",
            "cohort": "shrot",
            "setups": [
                {"setup_type": "pead", "score": 0.5, "trigger_level": 10.0, "stop_level": 11.0}
            ],
        }
    ]
    with pytest.raises(ValueError, match="cohort"):
        build_watchlist({"long": [], "short": []}, candidates, {}, watch_dsr_floor=0.0)


def test_build_watchlist_short_cohort_needs_short_tournament() -> None:
    # join-miss regression: a short-cohort candidate whose ticker ran only a
    # LONG tournament must produce no short watch row (and no long row either)
    tournament = {
        "long": [
            {
                "ticker": "ONLY",
                "stance": "long",
                "winner": None,
                "survivors": [],
                "verdicts": [{"strategy": "sma_cross", "deflated_sharpe": 0.9}],
            }
        ],
        "short": [],
    }
    candidates = [
        {
            "ticker": "ONLY",
            "cohort": "short",
            "setups": [
                {
                    "setup_type": "base_breakdown",
                    "score": 0.8,
                    "trigger_level": 50.0,
                    "stop_level": 55.0,
                }
            ],
        }
    ]
    watchlist = build_watchlist(tournament, candidates, {}, watch_dsr_floor=0.5)
    assert watchlist == {"long": [], "short": []}


def test_watch_row_carries_setup_entry_window() -> None:
    assert ENTRY_WINDOWS["pead"] == 15
    assert ENTRY_WINDOWS["pead_down"] == 15
    tournament = {
        "long": [
            {
                "ticker": "DRFT",
                "stance": "long",
                "winner": None,
                "survivors": [],
                "verdicts": [{"strategy": "sma_cross", "deflated_sharpe": 0.7}],
            }
        ],
        "short": [],
    }
    candidates = [
        {
            "ticker": "DRFT",
            "cohort": "long",
            "setups": [
                {"setup_type": "pead", "score": 0.9, "trigger_level": 100.0, "stop_level": 95.0}
            ],
        }
    ]
    watchlist = build_watchlist(tournament, candidates, {}, watch_dsr_floor=0.5)
    assert watchlist["long"][0]["entry_window"] == 15


def test_watch_row_default_entry_window_is_five() -> None:
    tournament = {
        "long": [
            {
                "ticker": "BASE",
                "stance": "long",
                "winner": None,
                "survivors": [],
                "verdicts": [{"strategy": "sma_cross", "deflated_sharpe": 0.7}],
            }
        ],
        "short": [],
    }
    candidates = [
        {
            "ticker": "BASE",
            "cohort": "long",
            "setups": [
                {
                    "setup_type": "base_breakout",
                    "score": 0.9,
                    "trigger_level": 100.0,
                    "stop_level": 95.0,
                }
            ],
        }
    ]
    watchlist = build_watchlist(tournament, candidates, {}, watch_dsr_floor=0.5)
    assert watchlist["long"][0]["entry_window"] == 5
