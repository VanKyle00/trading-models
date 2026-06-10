"""Tests for the /tournaments pages: view models, sparkline, and routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from webapp import scans as scans_module
from webapp import tournaments as tournaments_module


def _ledger_record(date: str) -> dict:
    return {
        "date": date,
        "ticker": "AAPL",
        "stance": "long",
        "strategy": "sma_cross",
        "levels": {"entry": 210.0, "entry_type": "market", "stop": 195.0, "target": 240.0},
        "status": "open",
        "entry_date": "2026-06-09",
        "entry_fill": 210.0,
        "exit_date": None,
        "exit_fill": None,
        "r": 0.6,
        "pct_move": 0.043,
        "sessions_held": 2,
        "ambiguous_bar": False,
        "closes": [["2026-06-09", 212.0], ["2026-06-10", 219.0]],
    }


def _ledger(date: str) -> dict:
    return {
        "built_asof": "2026-06-10",
        "stats": {
            "issued": 1,
            "waiting": 0,
            "expired": 0,
            "open": 1,
            "stopped": 0,
            "target": 0,
            "errors": 0,
            "hit_rate": None,
            "total_r": 0.6,
            "avg_r": 0.6,
        },
        "tickets": [_ledger_record(date)],
    }


def test_sparkline_needs_two_closes() -> None:
    assert tournaments_module.sparkline_svg({}) == ""
    assert tournaments_module.sparkline_svg({"closes": [["2026-06-09", 100.0]]}) == ""


def test_sparkline_draws_price_and_level_rules() -> None:
    svg = tournaments_module.sparkline_svg(_ledger_record("2026-06-08"))
    assert svg.startswith("<svg")
    assert "<polyline" in svg
    assert 'class="sl-entry"' in svg
    assert 'class="sl-stop"' in svg
    assert 'class="sl-target"' in svg


def test_sparkline_flat_data_is_empty() -> None:
    record = {"closes": [["2026-06-09", 100.0], ["2026-06-10", 100.0]], "levels": {}}
    assert tournaments_module.sparkline_svg(record) == ""


def test_ledger_rows_attach_sparklines() -> None:
    rows = tournaments_module.ledger_rows(_ledger("2026-06-08"))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["spark"].startswith("<svg")
    assert tournaments_module.ledger_rows(None) == []


def test_ledger_rows_normalize_error_records() -> None:
    ledger = {
        "stats": {},
        "tickets": [
            {
                "date": "2026-06-08",
                "ticker": "MISS",
                "stance": "short",
                "strategy": "sma_cross",
                "status": "error",
                "error": "no data for MISS",
            }
        ],
    }
    rows = tournaments_module.ledger_rows(ledger)
    assert rows[0]["r"] is None
    assert rows[0]["pct_move"] is None
    assert rows[0]["sessions_held"] == 0
    assert rows[0]["ambiguous_bar"] is False
    assert rows[0]["closes"] == []
    assert rows[0]["spark"] == ""


def test_catalog_summarizes_funnels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    base = tmp_path / "scans"
    (base / "2026-06-08").mkdir(parents=True)
    (base / "2026-06-08" / "report.json").write_text(
        json.dumps(
            {
                "asof": "2026-06-08",
                "funnel": {
                    "universe": 1003,
                    "fa_shortlist": 40,
                    "fa_shortlist_short": 40,
                    "with_setups": 1,
                    "tournament_candidates": 80,
                    "tournament_survivors": 1,
                    "tickets": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = tournaments_module.catalog(["2026-06-08", "9999-99-99"])
    assert rows == [
        {
            "date": "2026-06-08",
            "universe": 1003,
            "fa_long": 40,
            "fa_short": 40,
            "candidates": 80,
            "survivors": 1,
            "tickets": 1,
        }
    ]


def test_day_view_joins_outcomes_to_tickets() -> None:
    scan = {
        "asof": "2026-06-08",
        "funnel": {"universe": 1003},
        "fa_candidates": {"long": [], "short": []},
        "tournament": {
            "long": [
                {
                    "ticker": "AAPL",
                    "stance": "long",
                    "winner": {"strategy": "sma_cross"},
                    "winner_changed": False,
                    "survivors": ["sma_cross"],
                    "n_trials": 18,
                    "verdicts": [],
                }
            ],
            "short": [
                {
                    "ticker": "WEAK",
                    "stance": "short",
                    "winner": None,
                    "winner_changed": None,
                    "survivors": [],
                    "n_trials": 18,
                    "verdicts": [],
                }
            ],
        },
        "tickets": {
            "long": [{"ticker": "AAPL", "stance": "long", "strategy": "sma_cross"}],
            "short": [],
        },
    }
    day = tournaments_module.day_view(scan, _ledger("2026-06-08"))
    assert day["has_tournament"] is True
    assert day["n_survivor_tickers"] == 1
    assert len(day["tournament"]) == 2
    assert day["has_tickets"] is True
    outcome = day["tickets"]["long"][0]["outcome"]
    assert outcome["status"] == "open"
    assert outcome["spark"].startswith("<svg")


def test_day_view_without_ledger_or_tournament() -> None:
    scan = {"asof": "2026-06-07", "funnel": {"universe": 503}}
    day = tournaments_module.day_view(scan, None)
    assert day["has_tournament"] is False
    assert day["fa_candidates"] is None
    assert day["has_tickets"] is False
    assert day["tickets"] == {"long": [], "short": []}
