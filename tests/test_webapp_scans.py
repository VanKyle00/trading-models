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
        "funnel": {"universe": 503, "fa_shortlist": 40, "with_setups": 2},
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
        ],
        "errors": [{"ticker": "BAD", "stage": "bars", "error": "no data"}],
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


def test_index_links_to_scans(scan_dir: Path) -> None:
    client = TestClient(create_app())
    assert 'href="/scans"' in client.get("/").text
