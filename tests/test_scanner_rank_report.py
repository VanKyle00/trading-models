"""Tests for ranking and report writing (Stage 4 of the scanner)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradinglib.scanner.rank import rank_candidates
from tradinglib.scanner.report import load_latest_report, write_report


def _candidate(ticker: str, **overrides: object) -> dict:
    base: dict = {
        "ticker": ticker,
        "name": ticker,
        "sector": "Tech",
        "fa_score": 0.8,
        "setup_score": 0.6,
        "setups": [
            {
                "setup_type": "base_breakout",
                "score": 0.6,
                "trigger_level": 100.0,
                "stop_level": 92.0,
                "evidence": {"atr": 2.0},
            }
        ],
        "qualitative_score": None,
        "stance": None,
        "red_flags": [],
        "earnings_warning": False,
    }
    base.update(overrides)
    return base


def test_rank_orders_by_final_score() -> None:
    ranked = rank_candidates(
        [
            _candidate("LOW", fa_score=0.2, setup_score=0.2),
            _candidate("HIGH", fa_score=0.9, setup_score=0.9),
        ]
    )

    assert [c["ticker"] for c in ranked] == ["HIGH", "LOW"]
    assert ranked[0]["final_score"] > ranked[1]["final_score"]


def test_rank_renormalizes_without_qualitative() -> None:
    ranked = rank_candidates([_candidate("A", fa_score=0.8, setup_score=0.6)])

    expected = (0.35 * 0.8 + 0.45 * 0.6) / 0.80
    assert ranked[0]["final_score"] == pytest.approx(expected)


def test_rank_blends_qualitative_when_present() -> None:
    ranked = rank_candidates(
        [_candidate("A", fa_score=0.8, setup_score=0.6, qualitative_score=7.0)]
    )

    expected = 0.35 * 0.8 + 0.45 * 0.6 + 0.20 * 0.7
    assert ranked[0]["final_score"] == pytest.approx(expected)


def test_rank_pins_avoid_stance_to_bottom() -> None:
    ranked = rank_candidates(
        [
            _candidate("MEH", fa_score=0.3, setup_score=0.3),
            _candidate("TRAP", fa_score=0.95, setup_score=0.95, stance="avoid"),
        ]
    )

    assert [c["ticker"] for c in ranked] == ["MEH", "TRAP"]
    assert ranked[1]["pinned"]
    assert "avoid" in ranked[1]["pinned_reason"]


def test_rank_pins_red_flags_to_bottom() -> None:
    ranked = rank_candidates(
        [
            _candidate("OK"),
            _candidate("FLAGGED", fa_score=0.99, red_flags=["going concern doubt"]),
        ]
    )

    assert ranked[-1]["ticker"] == "FLAGGED"
    assert "going concern doubt" in ranked[-1]["pinned_reason"]


def test_rank_truncates_to_top() -> None:
    ranked = rank_candidates([_candidate(f"T{i}", fa_score=i / 10) for i in range(10)], top=3)

    assert len(ranked) == 3


def _result() -> dict:
    return {
        "asof": "2026-06-09",
        "config": {"fa_keep": 40, "top": 15},
        "funnel": {"universe": 503, "fa_shortlist": 40, "with_setups": 2},
        "candidates": rank_candidates([_candidate("AAPL"), _candidate("MSFT", fa_score=0.5)]),
        "errors": [{"ticker": "BAD", "stage": "bars", "error": "no data"}],
    }


def test_write_report_creates_json_and_md(tmp_path: Path) -> None:
    json_path, md_path = write_report(_result(), tmp_path)

    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["funnel"]["universe"] == 503
    assert {c["ticker"] for c in loaded["candidates"]} == {"AAPL", "MSFT"}

    md = md_path.read_text(encoding="utf-8")
    assert "AAPL" in md
    assert "503" in md  # funnel counts surface in the header
    assert "base_breakout" in md
    assert "BAD" in md  # errors are reported, not swallowed


def _structure(**overrides: object) -> dict:
    base: dict = {
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
    base.update(overrides)
    return base


def _ticket(ticker: str = "AAPL", stance: str = "long", **overrides: object) -> dict:
    base: dict = {
        "ticker": ticker,
        "stance": stance,
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
            "winner_changed": False,
            "fa_rank": 1,
            "fa_score": 0.9,
        },
        "quotes_asof": "2026-06-10",
        "iv_ratio": 1.05,
        "next_earnings": "2026-07-30",
        "account_size": 100000.0,
        "risk_per_trade_pct": 0.01,
        "structures": [_structure()],
        "warnings": ["indicative quotes: last/close marks as of quotes_asof"],
    }
    base.update(overrides)
    return base


def _ticket_result() -> dict:
    result = _result()
    result["funnel"].update({"tournament_candidates": 4, "tournament_survivors": 2, "tickets": 2})
    result["tickets"] = {
        "long": [_ticket()],
        "short": [
            _ticket(
                ticker="WEAK",
                stance="short",
                structures=[
                    _structure(
                        kind="stock_short",
                        label="short shares; stop 52.00",
                        max_loss=2.0,
                        quantity=None,
                        warnings=["borrow/squeeze risk; cost not modeled"],
                    ),
                    _structure(
                        kind="long_put",
                        label="buy 1x 50P 2026-07-17",
                        unit="contract",
                        max_loss=3.1,
                        max_gain=46.9,
                        pop_market_implied=0.42,
                        recommended=False,
                        quantity=3,
                    ),
                ],
                evidence={
                    "deflated_sharpe": 0.91,
                    "sharpe": 0.9,
                    "n_trades": 14,
                    "n_windows": 6,
                    "winner_changed": True,
                    "fa_rank": 2,
                    "fa_score": 0.1,
                },
            )
        ],
    }
    return result


def test_render_markdown_tickets_section(tmp_path: Path) -> None:
    _, md_path = write_report(_ticket_result(), tmp_path)
    md = md_path.read_text(encoding="utf-8")

    assert "## Tickets" in md
    assert "4 candidates" in md and "2 tickets" in md  # tournament funnel line
    assert "| AAPL | sma_cross | 210.00 | 195.00 | 240.00 |" in md  # table row
    assert "buy shares; stop 195.00" in md  # recommended structure label
    assert "66 shares" in md  # sized quantity
    assert "UNSIZED" in md  # zero-quantity structure is flagged, not hidden
    assert "winner changed" in md
    assert "indicative" in md.lower()  # quotes framing survives into the report
    assert "42%" in md  # PoP rendered as a percent
    assert "borrow" in md.lower()  # short-side borrow-cost footnote


def test_render_markdown_without_tickets_keys_is_unchanged(tmp_path: Path) -> None:
    _, md_path = write_report(_result(), tmp_path)  # pre-C report shape
    md = md_path.read_text(encoding="utf-8")

    assert "## Tickets" not in md
    assert "Tournament:" not in md
    assert "AAPL" in md  # classic watchlist still renders


def test_load_latest_report_returns_most_recent_prior(tmp_path: Path) -> None:
    for day, marker in (("2026-06-07", "old"), ("2026-06-08", "newer")):
        d = tmp_path / day
        d.mkdir()
        (d / "report.json").write_text(json.dumps({"asof": day, "marker": marker}))
    (tmp_path / "2026-06-09").mkdir()  # today's dir exists but has no report yet

    out = load_latest_report(tmp_path, before="2026-06-09")

    assert out is not None and out["marker"] == "newer"


def test_load_latest_report_excludes_today_and_handles_missing(tmp_path: Path) -> None:
    d = tmp_path / "2026-06-09"
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"asof": "2026-06-09"}))

    assert load_latest_report(tmp_path, before="2026-06-09") is None  # strictly before
    assert load_latest_report(tmp_path / "missing", before="2026-06-09") is None


def test_load_latest_report_swallows_corrupt_json(tmp_path: Path) -> None:
    d = tmp_path / "2026-06-08"
    d.mkdir()
    (d / "report.json").write_text('{"asof": "2026-06-08"')  # truncated partial write

    assert load_latest_report(tmp_path, before="2026-06-09") is None
