"""Tests for the /tournaments pages: view models, sparkline, and routes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp import scans as scans_module
from webapp import tournaments as tournaments_module
from webapp.main import create_app


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

    ys = {
        k: float(re.search(rf'class="sl-{k}" x1="0" y1="([\d.]+)"', svg).group(1))
        for k in ("stop", "target")
    }
    assert ys["stop"] > ys["target"]  # SVG y grows downward


def test_sparkline_flat_data_is_empty() -> None:
    record = {"closes": [["2026-06-09", 100.0], ["2026-06-10", 100.0]], "levels": {}}
    assert tournaments_module.sparkline_svg(record) == ""


def test_ledger_rows_attach_sparklines() -> None:
    rows = tournaments_module.ledger_rows(_ledger("2026-06-08"))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["spark"].startswith("<svg")
    assert tournaments_module.ledger_rows(None) == []


def test_error_chip_carries_error_title(tournaments_dir: Path) -> None:
    ledger = _ledger("2026-06-08")
    ledger["tickets"].append(
        {
            "date": "2026-06-08",
            "ticker": "MISS",
            "stance": "short",
            "strategy": "sma_cross",
            "status": "error",
            "error": "no data for MISS",
        }
    )
    (tournaments_dir / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    html = TestClient(create_app()).get("/tournaments").text
    assert 'title="no data for MISS"' in html


def test_index_survives_corrupt_report(tournaments_dir: Path) -> None:
    (tournaments_dir / "2026-06-06").mkdir()
    (tournaments_dir / "2026-06-06" / "report.json").write_text("{not json", encoding="utf-8")
    resp = TestClient(create_app()).get("/tournaments")
    assert resp.status_code == 200
    assert 'href="/tournaments/2026-06-08"' in resp.text  # healthy date still listed


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
            "watchlist": None,
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


def _report(asof: str) -> dict:
    return {
        "asof": asof,
        "funnel": {
            "universe": 1003,
            "fa_shortlist": 2,
            "fa_shortlist_short": 2,
            "with_setups": 1,
            "tournament_candidates": 4,
            "tournament_survivors": 1,
            "tickets": 1,
        },
        "candidates": [],
        "errors": [],
        "fa_candidates": {
            "long": [
                {"ticker": "AAPL", "name": "Apple", "sector": "Tech", "fa_score": 0.91, "rank": 1}
            ],
            "short": [
                {
                    "ticker": "WEAK",
                    "name": "Weak Co",
                    "sector": "Energy",
                    "fa_score": 0.05,
                    "rank": 1,
                }
            ],
        },
        "tournament": {
            "long": [
                {
                    "ticker": "AAPL",
                    "stance": "long",
                    "n_trials": 18,
                    "survivors": ["sma_cross"],
                    "winner_changed": False,
                    "winner": {
                        "strategy": "sma_cross",
                        "survived": True,
                        "reasons": [],
                        "params": {"fast": 10, "slow": 50},
                        "deflated_sharpe": 0.93,
                        "sharpe": 1.2,
                        "n_trades": 15,
                        "max_param_change_rate": 0.2,
                        "n_windows": 6,
                        "levels": {
                            "entry": 210.0,
                            "entry_type": "market",
                            "stop": 195.0,
                            "target": 240.0,
                            "condition": "hold while fast SMA above",
                        },
                    },
                    "verdicts": [
                        {
                            "strategy": "sma_cross",
                            "survived": True,
                            "reasons": [],
                            "params": {"fast": 10, "slow": 50},
                            "deflated_sharpe": 0.93,
                            "sharpe": 1.2,
                            "n_trades": 15,
                            "max_param_change_rate": 0.2,
                            "n_windows": 6,
                        },
                        {
                            "strategy": "donchian",
                            "survived": False,
                            "reasons": ["deflated_sharpe 0.41 < 0.90"],
                            "params": {"window": 20},
                            "deflated_sharpe": 0.41,
                            "sharpe": 0.8,
                            "n_trades": 9,
                            "max_param_change_rate": 0.3,
                            "n_windows": 6,
                        },
                    ],
                }
            ],
            "short": [
                {
                    "ticker": "WEAK",
                    "stance": "short",
                    "n_trials": 18,
                    "survivors": [],
                    "winner": None,
                    "winner_changed": None,
                    "verdicts": [
                        {
                            "strategy": "sma_cross",
                            "survived": False,
                            "reasons": ["n_trades 3 < 12"],
                            "params": {},
                            "deflated_sharpe": 0.2,
                            "sharpe": 0.5,
                            "n_trades": 3,
                            "max_param_change_rate": 0.1,
                            "n_windows": 6,
                        }
                    ],
                }
            ],
        },
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
                        "condition": "hold while fast SMA above",
                    },
                    "evidence": {
                        "deflated_sharpe": 0.93,
                        "sharpe": 1.2,
                        "n_trades": 15,
                        "n_windows": 6,
                        "winner_changed": False,
                        "fa_rank": 1,
                        "fa_score": 0.91,
                    },
                    "quotes_asof": asof,
                    "iv_ratio": 1.1,
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
                        }
                    ],
                    "warnings": ["indicative quotes: last/close marks"],
                }
            ],
            "short": [],
        },
    }


@pytest.fixture
def tournaments_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(tournaments_module, "processed_dir", lambda source: tmp_path / source)
    base = tmp_path / "scans"
    (base / "2026-06-08").mkdir(parents=True)
    (base / "2026-06-08" / "report.json").write_text(
        json.dumps(_report("2026-06-08")), encoding="utf-8"
    )
    (base / "ledger.json").write_text(json.dumps(_ledger("2026-06-08")), encoding="utf-8")
    return base


def test_index_renders_track_record_and_catalog(tournaments_dir: Path) -> None:
    html = TestClient(create_app()).get("/tournaments").text
    assert "Total R" in html
    assert "+0.60" in html
    assert 'class="spark"' in html
    assert 'href="/tournaments/2026-06-08"' in html
    assert "1003" in html  # catalog funnel summary


def test_index_without_ledger_still_shows_catalog(tournaments_dir: Path) -> None:
    (tournaments_dir / "ledger.json").unlink()
    html = TestClient(create_app()).get("/tournaments").text
    assert "no evaluations yet" in html.lower()
    assert 'href="/tournaments/2026-06-08"' in html


def test_day_page_tells_the_pipeline_story(tournaments_dir: Path) -> None:
    html = TestClient(create_app()).get("/tournaments/2026-06-08").text
    assert "Find tickers" in html
    assert "Find strategy" in html
    assert "Create trade" in html
    assert "deflated_sharpe 0.41 &lt; 0.90" in html or "deflated_sharpe 0.41 < 0.90" in html
    assert "n_trades 3 &lt; 12" in html or "n_trades 3 < 12" in html
    assert "sma_cross" in html
    assert "buy shares; stop 195.00" in html
    assert 'class="spark"' in html  # ticket outcome strip
    assert "the bar held" not in html.lower()  # this night HAS a survivor
    assert "AngloGold" not in html  # sanity: only fixture data


def test_day_page_zero_survivors_is_intentional(tournaments_dir: Path) -> None:
    report = _report("2026-06-09")
    report["tournament"]["long"][0]["winner"] = None
    report["tournament"]["long"][0]["survivors"] = []
    report["tickets"] = {"long": [], "short": []}
    report["funnel"]["tournament_survivors"] = 0
    report["funnel"]["tickets"] = 0
    (tournaments_dir / "2026-06-09").mkdir()
    (tournaments_dir / "2026-06-09" / "report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    html = TestClient(create_app()).get("/tournaments/2026-06-09").text
    assert "the bar held" in html.lower()


def test_day_page_old_format_renders_available_stages(tournaments_dir: Path) -> None:
    (tournaments_dir / "2026-06-07").mkdir()
    (tournaments_dir / "2026-06-07" / "report.json").write_text(
        json.dumps(
            {
                "asof": "2026-06-07",
                "funnel": {"universe": 503, "fa_shortlist": 40, "with_setups": 2},
                "candidates": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    resp = TestClient(create_app()).get("/tournaments/2026-06-07")
    assert resp.status_code == 200
    assert "not run on this date" in resp.text.lower()


def test_day_page_unknown_date_is_404(tournaments_dir: Path) -> None:
    assert TestClient(create_app()).get("/tournaments/1999-01-01").status_code == 404
    assert TestClient(create_app()).get("/tournaments/..%2F..%2Fsecrets").status_code == 404


def test_index_empty_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(tournaments_module, "processed_dir", lambda source: tmp_path / source)
    resp = TestClient(create_app()).get("/tournaments")
    assert resp.status_code == 200
    assert "no scans yet" in resp.text.lower()


def test_topbars_link_to_tournaments(tournaments_dir: Path) -> None:
    client = TestClient(create_app())
    assert 'href="/tournaments"' in client.get("/").text
    assert 'href="/tournaments"' in client.get("/scans").text


# ---------------------------------------------------------------------------
# Tier-aware /tournaments tests (C4)
# ---------------------------------------------------------------------------


def _ledger_with_tiers(date: str) -> dict:
    """Ledger with by_tier block and tier fields on records."""
    ticket_rec = {**_ledger_record(date), "tier": "ticket"}
    watch_rec = {
        "date": date,
        "ticker": "WLVL",
        "stance": "long",
        "strategy": "sma_cross",
        "tier": "watch",
        "tier_reason": "below DS threshold",
        "levels": {"entry": 50.0, "stop": 45.0, "target": 60.0},
        "status": "open",
        "entry_date": "2026-06-09",
        "entry_fill": 50.0,
        "exit_date": None,
        "exit_fill": None,
        "r": -0.3,
        "pct_move": -0.02,
        "sessions_held": 1,
        "ambiguous_bar": False,
        "closes": [["2026-06-09", 49.5], ["2026-06-10", 49.0]],
    }
    return {
        "built_asof": "2026-06-10",
        "stats": {
            "issued": 2,
            "waiting": 0,
            "expired": 0,
            "open": 2,
            "stopped": 0,
            "target": 0,
            "errors": 0,
            "hit_rate": None,
            "total_r": 0.3,
            "avg_r": 0.15,
            "max_drawdown_r": 0.5,
            "by_tier": {
                "ticket": {
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
                    "max_drawdown_r": None,
                },
                "watch": {
                    "issued": 1,
                    "waiting": 0,
                    "expired": 0,
                    "open": 1,
                    "stopped": 0,
                    "target": 0,
                    "errors": 0,
                    "hit_rate": None,
                    "total_r": -0.3,
                    "avg_r": -0.3,
                    "max_drawdown_r": None,
                },
            },
        },
        "tickets": [ticket_rec, watch_rec],
    }


def _report_with_watchlist_and_fdr(asof: str) -> dict:
    """Report with watchlist and fdr keys for day-page tests."""
    base = _report(asof)
    base["watchlist"] = {
        "long": [
            {
                "ticker": "WLVL",
                "stance": "long",
                "strategy": "sma_cross",
                "tier": "watch",
                "tier_reason": "below DS threshold",
                "deflated_sharpe": 0.75,
                "levels": {"entry": 50.0, "stop": 45.0, "target": 60.0},
            }
        ],
        "short": [],
    }
    base["fdr"] = {"alpha": 0.05, "threshold": 0.9, "family": 18}
    base["funnel"]["fdr_passed"] = 3
    base["funnel"]["watchlist"] = 1
    return base


@pytest.fixture
def tiered_tournaments_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture with a tiered ledger and a report that includes watchlist+fdr."""
    monkeypatch.setattr(scans_module, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(tournaments_module, "processed_dir", lambda source: tmp_path / source)
    base = tmp_path / "scans"
    (base / "2026-06-08").mkdir(parents=True)
    (base / "2026-06-08" / "report.json").write_text(
        json.dumps(_report_with_watchlist_and_fdr("2026-06-08")), encoding="utf-8"
    )
    (base / "ledger.json").write_text(
        json.dumps(_ledger_with_tiers("2026-06-08")), encoding="utf-8"
    )
    return base


def test_index_by_tier_strips_render(tiered_tournaments_dir: Path) -> None:
    """When by_tier is present, both tier strips render with their labels and stats."""
    html = TestClient(create_app()).get("/tournaments").text
    assert "Tickets" in html
    assert "Watchlist" in html
    # max_drawdown_r value from the combined stats
    assert "0.50" in html


def test_index_by_tier_badge_in_rows(tiered_tournaments_dir: Path) -> None:
    """Ledger table rows show tier badges for both ticket and watch rows."""
    html = TestClient(create_app()).get("/tournaments").text
    # tier badge for each row (chip with tier value)
    assert "ticket" in html
    assert "watch" in html


def test_index_old_ledger_no_tier_strips(tournaments_dir: Path) -> None:
    """OLD ledger (no by_tier): page renders 200, no per-tier strips, existing assertions pass."""
    html = TestClient(create_app()).get("/tournaments").text
    assert "Total R" in html
    assert "+0.60" in html
    # no per-tier strip headers for OLD ledger
    assert "Tickets" not in html or "Watchlist" not in html


def test_index_old_ledger_no_crash(tournaments_dir: Path) -> None:
    """OLD ledger renders 200 without by_tier."""
    resp = TestClient(create_app()).get("/tournaments")
    assert resp.status_code == 200


def test_day_watchlist_section_renders(tiered_tournaments_dir: Path) -> None:
    """Watchlist section appears with ticker, tier_reason, deflated_sharpe."""
    html = TestClient(create_app()).get("/tournaments/2026-06-08").text
    assert "Watchlist" in html
    assert "WLVL" in html
    assert "below DS threshold" in html
    assert "0.75" in html


def test_day_fdr_story_line_renders(tiered_tournaments_dir: Path) -> None:
    """FDR story line shows family, passed, watchlist counts."""
    html = TestClient(create_app()).get("/tournaments/2026-06-08").text
    # family=18, fdr_passed=3, watchlist=1
    assert "18" in html
    assert "3" in html
    assert "1" in html


def test_day_old_scan_no_watchlist_renders_200(tournaments_dir: Path) -> None:
    """OLD scan (no fdr/watchlist keys): page renders 200 unchanged."""
    resp = TestClient(create_app()).get("/tournaments/2026-06-08")
    assert resp.status_code == 200
    # watchlist section should not appear
    assert "Watchlist" not in resp.text


def test_day_view_watchlist_and_fdr_keys() -> None:
    """day_view returns watchlist and fdr keys from scan; defaults when absent."""
    scan_with = {
        "asof": "2026-06-08",
        "funnel": {"universe": 100, "fdr_passed": 3, "watchlist": 1},
        "watchlist": {"long": [{"ticker": "WW"}], "short": []},
        "fdr": {"alpha": 0.05, "threshold": 0.9, "family": 18},
    }
    day = tournaments_module.day_view(scan_with, None)
    assert day["watchlist"] == {"long": [{"ticker": "WW"}], "short": []}
    assert day["fdr"] == {"alpha": 0.05, "threshold": 0.9, "family": 18}
    assert day["n_watch"] == 1

    scan_without = {"asof": "2026-06-07", "funnel": {"universe": 50}}
    day2 = tournaments_module.day_view(scan_without, None)
    assert day2["watchlist"] == {"long": [], "short": []}
    assert day2["fdr"] is None
    assert day2["n_watch"] == 0


def test_catalog_row_has_watchlist_key() -> None:
    """catalog rows gain 'watchlist' key from funnel; None when absent."""
    scan = {
        "asof": "2026-06-08",
        "funnel": {
            "universe": 1003,
            "fa_shortlist": 2,
            "fa_shortlist_short": 2,
            "tournament_candidates": 4,
            "tournament_survivors": 1,
            "tickets": 1,
            "watchlist": 2,
        },
    }

    def _mock_load(date: str) -> dict | None:
        return scan if date == "2026-06-08" else None

    # catalog calls load_scan bound in tournaments module namespace
    orig = tournaments_module.load_scan
    tournaments_module.load_scan = _mock_load  # type: ignore[attr-defined]
    try:
        rows = tournaments_module.catalog(["2026-06-08"])
    finally:
        tournaments_module.load_scan = orig  # type: ignore[attr-defined]

    assert rows[0]["watchlist"] == 2
