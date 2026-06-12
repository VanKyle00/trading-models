"""Tests for the /scans pages (scan-report discovery + rendering)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp import scans as scans_module
from webapp.main import create_app


def _result(asof: str) -> dict:
    return {
        "asof": asof,
        "config": {"fa_keep": 40, "top": 15},
        "funnel": {
            "universe": 503,
            "fa_shortlist": 40,
            "with_setups": 2,
            "tournament_candidates": 4,
            "tournament_survivors": 1,
            "tickets": 1,
        },
        "candidates": [
            {
                "ticker": "AAPL",
                "name": "Apple",
                "sector": "Tech",
                "fa_score": 0.9,
                "setup_score": 0.7,
                "final_score": 0.79,
                "pinned": False,
                "pinned_reason": "",
                "qualitative_score": 8.0,
                "stance": "favorable",
                "red_flags": [],
                "earnings_warning": True,
                "setups": [
                    {
                        "setup_type": "base_breakout",
                        "score": 0.7,
                        "trigger_level": 210.0,
                        "stop_level": 195.0,
                        "evidence": {"atr": 2.0},
                    }
                ],
                "brief": {
                    "thesis": "Strong product cycle into the fall.",
                    "catalysts": ["WWDC"],
                    "risks": ["China demand"],
                    "red_flags": [],
                    "stance": "favorable",
                    "qualitative_score": 8.0,
                    "confidence": 0.7,
                    "parse_error": False,
                },
            },
            {
                "ticker": "TRAP",
                "name": "Trap Corp",
                "sector": "Energy",
                "fa_score": 0.95,
                "setup_score": 0.9,
                "final_score": 0.91,
                "pinned": True,
                "pinned_reason": "red flags: going concern doubt",
                "qualitative_score": 1.0,
                "stance": "avoid",
                "red_flags": ["going concern doubt"],
                "earnings_warning": False,
                "setups": [
                    {
                        "setup_type": "pead",
                        "score": 0.9,
                        "trigger_level": 50.0,
                        "stop_level": 45.0,
                        "evidence": {},
                    }
                ],
                "brief": None,
            },
            {
                "ticker": "WEAK",
                "name": "Weak Corp",
                "sector": "Energy",
                "fa_score": 0.1,
                "setup_score": 0.6,
                "final_score": 0.35,
                "pinned": False,
                "pinned_reason": "",
                "qualitative_score": None,
                "stance": None,
                "cohort": "short",
                "red_flags": [],
                "earnings_warning": False,
                "setups": [
                    {
                        "setup_type": "base_breakdown",
                        "score": 0.6,
                        "trigger_level": 48.0,
                        "stop_level": 52.0,
                        "evidence": {},
                    }
                ],
                "brief": None,
            },
        ],
        "errors": [{"ticker": "BAD", "stage": "bars", "error": "no data"}],
        "tickets": {
            "long": [
                {
                    "ticker": "AAPL",
                    "stance": "long",
                    "strategy": "sma_cross",
                    "style": "trend",
                    "params": {"fast": 10, "slow": 50},
                    "levels": {
                        "entry": 210.0,
                        "entry_type": "market",
                        "stop": 195.0,
                        "target": 240.0,
                        "condition": "enter while SMA(10) holds above SMA(50)",
                    },
                    "evidence": {
                        "deflated_sharpe": 0.93,
                        "sharpe": 1.1,
                        "n_trades": 18,
                        "n_windows": 6,
                        "winner_changed": True,
                        "fa_rank": 1,
                        "fa_score": 0.9,
                    },
                    "quotes_asof": "2026-06-10",
                    "iv_ratio": 1.05,
                    "next_earnings": None,
                    "account_size": 100000.0,
                    "risk_per_trade_pct": 0.01,
                    "structures": [
                        {
                            "kind": "stock",
                            "label": "buy shares; stop 195.00",
                            "unit": "share",
                            "legs": [],
                            "premium": None,
                            "max_loss": 15.0,
                            "max_gain": None,
                            "breakeven": 210.0,
                            "pop_market_implied": None,
                            "rr": 2.0,
                            "premium_yield": None,
                            "warnings": [],
                            "recommended": True,
                            "quantity": 66,
                            "loss_at_stop": None,
                        },
                        {
                            "kind": "long_call",
                            "label": "buy 1x 215C 2026-08-21",
                            "unit": "contract",
                            "legs": [],
                            "premium": 9.4,
                            "max_loss": 9.4,
                            "max_gain": None,
                            "breakeven": 224.4,
                            "pop_market_implied": 0.38,
                            "rr": None,
                            "premium_yield": None,
                            "warnings": [],
                            "recommended": False,
                            "quantity": None,
                            "loss_at_stop": None,
                        },
                    ],
                    "warnings": ["indicative quotes: last/close marks"],
                }
            ],
            "short": [],
        },
    }


@pytest.fixture
def scan_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    base = tmp_path / "scans"
    for asof in ("2026-06-08", "2026-06-09"):
        out = base / asof
        out.mkdir(parents=True)
        (out / "report.json").write_text(json.dumps(_result(asof)), encoding="utf-8")
    return base


def test_list_scan_dates_newest_first(scan_dir: Path) -> None:
    assert scans_module.list_scan_dates() == ["2026-06-09", "2026-06-08"]


def test_load_scan_rejects_bad_date_strings(scan_dir: Path) -> None:
    assert scans_module.load_scan("../../secrets") is None
    assert scans_module.load_scan("2026-06-09") is not None


def test_load_scan_corrupt_report_returns_none(scan_dir: Path) -> None:
    out = scan_dir / "2026-06-10"
    out.mkdir()
    (out / "report.json").write_text("{not json", encoding="utf-8")
    assert scans_module.load_scan("2026-06-10") is None


def test_scans_page_renders_latest(scan_dir: Path) -> None:
    client = TestClient(create_app())
    resp = client.get("/scans")

    assert resp.status_code == 200
    html = resp.text
    assert "AAPL" in html
    assert "503" in html  # funnel counts
    assert "base_breakout" in html
    assert "Strong product cycle" in html  # LLM thesis
    assert "2026-06-08" in html  # older scan is linked


def test_scans_page_shows_pin_and_warnings(scan_dir: Path) -> None:
    client = TestClient(create_app())
    html = client.get("/scans").text

    assert "going concern doubt" in html  # pinned reason surfaces
    assert "earnings" in html.lower()  # earnings-soon warning surfaces


def test_scans_page_shows_short_cohort_chip(scan_dir: Path) -> None:
    client = TestClient(create_app())
    html = client.get("/scans").text

    # WEAK is a short-cohort candidate — the card head must render the chip
    assert "WEAK" in html
    assert 'chip flag">short</span>' in html


def test_scans_page_specific_date(scan_dir: Path) -> None:
    client = TestClient(create_app())
    resp = client.get("/scans/2026-06-08")

    assert resp.status_code == 200
    assert "2026-06-08" in resp.text


def test_scans_page_unknown_date_is_404(scan_dir: Path) -> None:
    client = TestClient(create_app())
    assert client.get("/scans/1999-01-01").status_code == 404


def test_scans_page_empty_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    client = TestClient(create_app())
    resp = client.get("/scans")

    assert resp.status_code == 200
    assert "no scans yet" in resp.text.lower()


def test_scans_page_renders_tickets(scan_dir: Path) -> None:
    client = TestClient(create_app())
    html = client.get("/scans").text

    assert "Trade tickets" in html
    assert "sma_cross" in html
    assert "buy shares; stop 195.00" in html  # recommended structure
    assert "UNSIZED" in html  # null quantity is surfaced, not hidden
    assert "winner changed" in html
    assert "indicative" in html.lower()
    assert "Tournament" in html  # funnel metrics extended


def test_scans_page_without_ticket_keys_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reports written before sub-project C have no tickets/tournament keys."""
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    legacy = _result("2026-06-01")
    del legacy["tickets"]
    legacy["funnel"] = {"universe": 503, "fa_shortlist": 40, "with_setups": 2}
    out = tmp_path / "scans" / "2026-06-01"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(legacy), encoding="utf-8")

    client = TestClient(create_app())
    resp = client.get("/scans")

    assert resp.status_code == 200
    assert "Trade tickets" not in resp.text


def _promoted_ticket() -> dict:
    """A pooled-certified promoted ticket: NONE of build_ticket's fields.

    Shape matches tradinglib.scanner.pipeline's promotion of a setup watch row:
    no structures/evidence/style, just levels + pooled_evidence.
    """
    return {
        "ticker": "PEAD",
        "stance": "long",
        "strategy": "setup:pead",
        "tier": "ticket",
        "tier_reason": "pooled-certified",
        "entry_window": 7,
        "deflated_sharpe": 0.74,
        "levels": {
            "entry": 120.0,
            "entry_type": "stop",
            "stop": 110.0,
            "target": 140.0,
        },
        "pooled_evidence": {"pooled_dsr": 0.965, "n_dates": 24, "total_r": 6.1},
    }


def test_scans_page_renders_promoted_pooled_ticket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pooled-certified promoted ticket (no structures/evidence) must render, not 500."""
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    report = _result("2026-06-02")
    report["tickets"]["long"].append(_promoted_ticket())
    out = tmp_path / "scans" / "2026-06-02"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")

    client = TestClient(create_app())
    resp = client.get("/scans/2026-06-02")

    assert resp.status_code == 200
    html = resp.text
    assert "PEAD" in html
    assert "pooled-certified" in html
    assert "setup:pead" in html
    assert "120.00" in html  # levels still render


def test_index_links_to_scans(scan_dir: Path) -> None:
    client = TestClient(create_app())
    assert 'href="/scans"' in client.get("/").text
