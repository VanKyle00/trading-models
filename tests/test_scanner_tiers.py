"""Tests for nightly FDR tiering over tournament results."""

from __future__ import annotations

from tradinglib.scanner.tiers import apply_fdr, build_watchlist


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
