"""Tests for the forward ticket-performance ledger (build/write/load)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tradinglib.scanner.ledger import build_ledger, load_ledger, write_ledger


def _ticket(ticker: str, entry: float = 100.0, entry_type: str = "market") -> dict:
    return {
        "ticker": ticker,
        "stance": "long",
        "strategy": "sma_cross",
        "levels": {
            "entry": entry,
            "entry_type": entry_type,
            "stop": entry - 5.0,
            "target": entry + 10.0,
            "condition": "test",
        },
    }


def _report(asof: str, long: list[dict], short: list[dict]) -> dict:
    return {
        "asof": asof,
        "funnel": {},
        "candidates": [],
        "errors": [],
        "tickets": {"long": long, "short": short},
    }


@pytest.fixture
def base(tmp_path: Path) -> Path:
    root = tmp_path / "scans"
    # corrupt report: skipped, never fatal
    (root / "2026-06-05").mkdir(parents=True)
    (root / "2026-06-05" / "report.json").write_text("{not json", encoding="utf-8")
    # two tickets: TST resolves, MISS has no bar data
    (root / "2026-06-08").mkdir()
    (root / "2026-06-08" / "report.json").write_text(
        json.dumps(_report("2026-06-08", [_ticket("TST")], [_ticket("MISS")])),
        encoding="utf-8",
    )
    # TST re-issued the next night with a far-away stop trigger -> waiting
    (root / "2026-06-09").mkdir()
    (root / "2026-06-09" / "report.json").write_text(
        json.dumps(_report("2026-06-09", [_ticket("TST", entry=200.0, entry_type="stop")], [])),
        encoding="utf-8",
    )
    # old-format report (pre-tickets): contributes nothing
    (root / "2026-06-10").mkdir()
    (root / "2026-06-10" / "report.json").write_text(
        json.dumps({"asof": "2026-06-10", "funnel": {}, "candidates": [], "errors": []}),
        encoding="utf-8",
    )
    # non-date directory: ignored
    (root / "smoke-tickets").mkdir()
    (root / "smoke-tickets" / "report.json").write_text("{}", encoding="utf-8")
    return root


def _loader_with(bars_by_ticker: dict[str, pd.DataFrame]):
    calls: list[str] = []

    def loader(ticker: str, start: str | None = None, **kwargs: object) -> pd.DataFrame:
        calls.append(ticker)
        if ticker not in bars_by_ticker:
            raise RuntimeError(f"no data for {ticker}")
        return bars_by_ticker[ticker]

    loader.calls = calls  # type: ignore[attr-defined]
    return loader


def _tst_bars() -> pd.DataFrame:
    # 2026-06-09 (Tue): entry at open 100; 2026-06-10 (Wed): high 111 >= target 110
    idx = pd.date_range("2026-06-09", periods=2, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 109.0],
            "high": [101.0, 111.0],
            "low": [99.0, 108.0],
            "close": [100.0, 110.5],
        },
        index=idx,
    )


def test_build_scores_every_ticket_across_dates(base: Path) -> None:
    loader = _loader_with({"TST": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)

    assert ledger["built_asof"] == "2026-06-11"
    assert ledger["stats"]["issued"] == 3
    assert ledger["stats"]["target"] == 1
    assert ledger["stats"]["waiting"] == 1
    assert ledger["stats"]["errors"] == 1
    assert ledger["stats"]["hit_rate"] == pytest.approx(1.0)
    assert ledger["stats"]["total_r"] == pytest.approx(2.0)
    assert ledger["stats"]["avg_r"] == pytest.approx(2.0)


def test_records_are_newest_first_and_carry_identity(base: Path) -> None:
    ledger = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    first = ledger["tickets"][0]
    assert first["date"] == "2026-06-09"
    assert {"ticker", "stance", "strategy", "levels", "status"} <= set(first)
    assert first["levels"] == {"entry": 200.0, "entry_type": "stop", "stop": 195.0, "target": 210.0}


def test_loader_failures_become_error_records(base: Path) -> None:
    ledger = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    errors = [r for r in ledger["tickets"] if r["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["ticker"] == "MISS"
    assert "no data" in errors[0]["error"]


def test_loader_called_once_per_ticker(base: Path) -> None:
    loader = _loader_with({"TST": _tst_bars()})
    build_ledger(base, asof="2026-06-11", loader=loader)
    assert loader.calls.count("TST") == 1  # type: ignore[attr-defined]
    assert loader.calls.count("MISS") == 1  # type: ignore[attr-defined]


def test_write_and_load_round_trip(base: Path) -> None:
    ledger = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    path = write_ledger(ledger, base)
    assert path.name == "ledger.json"
    assert load_ledger(base) == ledger


def test_load_missing_or_corrupt_returns_none(base: Path) -> None:
    assert load_ledger(base) is None
    (base / "ledger.json").write_text("{nope", encoding="utf-8")
    assert load_ledger(base) is None


def test_empty_base_builds_empty_ledger(tmp_path: Path) -> None:
    ledger = build_ledger(tmp_path / "absent", asof="2026-06-11", loader=_loader_with({}))
    assert ledger["stats"]["issued"] == 0
    assert ledger["tickets"] == []


def test_loader_asked_for_fresh_bars(base: Path) -> None:
    # the parquet cache is frozen at issue night in prod; without refresh the
    # ledger would never see post-issue bars and every ticket waits forever
    seen_kwargs: list[dict] = []
    bars = {"TST": _tst_bars()}

    def loader(ticker: str, start: str | None = None, **kwargs: object) -> pd.DataFrame:
        seen_kwargs.append(dict(kwargs))
        if ticker not in bars:
            raise RuntimeError(f"no data for {ticker}")
        return bars[ticker]

    build_ledger(base, asof="2026-06-11", loader=loader)
    assert seen_kwargs and all(k.get("refresh") is True for k in seen_kwargs)


def test_malformed_ticket_becomes_error_record(base: Path) -> None:
    broken = {"ticker": "BRK", "stance": "long", "strategy": "sma_cross", "levels": {}}
    (base / "2026-06-09" / "report.json").write_text(
        json.dumps(_report("2026-06-09", [broken], [])), encoding="utf-8"
    )
    ledger = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    errors = [r for r in ledger["tickets"] if r["status"] == "error"]
    assert {"BRK", "MISS"} <= {r["ticker"] for r in errors}


# ---------------------------------------------------------------------------
# Tiered ledger tests (C3)
# ---------------------------------------------------------------------------


def _watch_row(ticker: str, entry: float = 100.0, with_levels: bool = True) -> dict:
    row: dict = {
        "ticker": ticker,
        "stance": "long",
        "strategy": "sma_cross",
        "tier": "watch",
        "tier_reason": "below DS threshold",
        "deflated_sharpe": 0.75,
    }
    if with_levels:
        row["levels"] = {
            "entry": entry,
            "entry_type": "market",
            "stop": entry - 5.0,
            "target": entry + 10.0,
        }
    else:
        row["levels"] = None
    return row


def _report_with_watchlist(
    asof: str,
    long: list[dict],
    short: list[dict],
    watchlist_long: list[dict] | None = None,
    watchlist_short: list[dict] | None = None,
) -> dict:
    return {
        "asof": asof,
        "funnel": {},
        "candidates": [],
        "errors": [],
        "tickets": {"long": long, "short": short},
        "watchlist": {
            "long": watchlist_long or [],
            "short": watchlist_short or [],
        },
    }


def _base_with_watchlist(tmp_path: Path) -> Path:
    root = tmp_path / "scans"
    (root / "2026-06-08").mkdir(parents=True)
    ticket = _ticket("TST")
    ticket["tier"] = "ticket"
    watch_with_levels = _watch_row("WLVL")
    watch_no_levels = _watch_row("WNOL", with_levels=False)
    (root / "2026-06-08" / "report.json").write_text(
        json.dumps(
            _report_with_watchlist(
                "2026-06-08",
                [ticket],
                [],
                watchlist_long=[watch_with_levels, watch_no_levels],
            )
        ),
        encoding="utf-8",
    )
    return root


def test_issues_includes_watchlist_rows_with_levels(tmp_path: Path) -> None:
    """_issues yields watch rows that have levels; skips those without."""
    from tradinglib.scanner.ledger import _issues

    base = _base_with_watchlist(tmp_path)
    issues = _issues(base)
    tickers = {t["ticker"] for _, t in issues}
    assert "TST" in tickers
    assert "WLVL" in tickers
    assert "WNOL" not in tickers  # no levels => skipped


def test_records_carry_tier_field(tmp_path: Path) -> None:
    """Every ledger record has a 'tier' key; old reports default to 'ticket'."""
    base = _base_with_watchlist(tmp_path)
    loader = _loader_with({"TST": _tst_bars(), "WLVL": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)
    tiers = {r["ticker"]: r["tier"] for r in ledger["tickets"]}
    assert tiers["TST"] == "ticket"
    assert tiers["WLVL"] == "watch"


def test_old_reports_default_tier_to_ticket(base: Path) -> None:
    """Reports without tier key on ticket dicts default to 'ticket'."""
    loader = _loader_with({"TST": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)
    for r in ledger["tickets"]:
        assert r["tier"] == "ticket"


def test_stats_has_by_tier(tmp_path: Path) -> None:
    """stats block has 'by_tier' with 'ticket' and 'watch' sub-blocks."""
    base = _base_with_watchlist(tmp_path)
    loader = _loader_with({"TST": _tst_bars(), "WLVL": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)
    by_tier = ledger["stats"]["by_tier"]
    assert "ticket" in by_tier
    assert "watch" in by_tier
    assert by_tier["ticket"]["issued"] == 1
    assert by_tier["watch"]["issued"] == 1


def test_stats_has_max_drawdown_r(tmp_path: Path) -> None:
    """stats block has 'max_drawdown_r'; None when no closed trades."""
    base = _base_with_watchlist(tmp_path)
    loader = _loader_with({"TST": _tst_bars(), "WLVL": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)
    # The stats block must carry the key
    assert "max_drawdown_r" in ledger["stats"]
    # by_tier sub-blocks too
    assert "max_drawdown_r" in ledger["stats"]["by_tier"]["ticket"]
    assert "max_drawdown_r" in ledger["stats"]["by_tier"]["watch"]


def test_max_drawdown_r_golden() -> None:
    """Golden path: rs [+2, -1, -1, +1] in exit order → max DD == 2.0."""
    from tradinglib.scanner.ledger import _max_drawdown_r

    records = [
        {
            "status": "target",
            "r": 2.0,
            "exit_date": "2026-06-01",
            "ticker": "A",
            "date": "2026-05-28",
        },
        {
            "status": "stopped",
            "r": -1.0,
            "exit_date": "2026-06-02",
            "ticker": "B",
            "date": "2026-05-28",
        },
        {
            "status": "stopped",
            "r": -1.0,
            "exit_date": "2026-06-03",
            "ticker": "C",
            "date": "2026-05-28",
        },
        {
            "status": "target",
            "r": 1.0,
            "exit_date": "2026-06-04",
            "ticker": "D",
            "date": "2026-05-28",
        },
    ]
    assert _max_drawdown_r(records) == pytest.approx(2.0)


def test_max_drawdown_r_none_when_no_closed() -> None:
    """Returns None when there are no closed trades."""
    from tradinglib.scanner.ledger import _max_drawdown_r

    records = [
        {"status": "open", "r": 0.5, "exit_date": None, "ticker": "A", "date": "2026-06-01"},
        {"status": "waiting", "r": None, "exit_date": None, "ticker": "B", "date": "2026-06-01"},
    ]
    assert _max_drawdown_r(records) is None


def test_watch_rows_simulate_like_tickets(tmp_path: Path) -> None:
    """Watch rows with levels are scored by simulate_ticket and appear in records."""
    base = _base_with_watchlist(tmp_path)
    loader = _loader_with({"TST": _tst_bars(), "WLVL": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)
    watch_records = [r for r in ledger["tickets"] if r["tier"] == "watch"]
    assert len(watch_records) == 1
    assert watch_records[0]["ticker"] == "WLVL"
    assert watch_records[0]["status"] in (
        "target",
        "stopped",
        "open",
        "waiting",
        "expired",
        "error",
    )


def test_stats_excludes_open_mtm_from_realized_r() -> None:
    """total_r/avg_r aggregate ONLY closed (target/stopped) trades; open marks
    are reported separately as open_unrealized_r so an unrealized position can
    never inflate the headline R used to compare models (audit A5)."""
    from tradinglib.scanner.ledger import _stats

    records = [
        {"status": "target", "r": 2.0, "ticker": "A", "date": "2026-06-01"},
        {"status": "stopped", "r": -1.0, "ticker": "B", "date": "2026-06-01"},
        {"status": "open", "r": 0.5, "ticker": "C", "date": "2026-06-01"},
        {"status": "waiting", "r": None, "ticker": "D", "date": "2026-06-01"},
    ]
    stats = _stats(records)
    assert stats["total_r"] == pytest.approx(1.0)  # 2.0 + (-1.0), NOT + 0.5 open
    assert stats["avg_r"] == pytest.approx(0.5)  # over the 2 closed trades only
    assert stats["open_unrealized_r"] == pytest.approx(0.5)  # the open mark, segregated
    assert stats["hit_rate"] == pytest.approx(0.5)  # already closed-only, unchanged


def test_stats_open_unrealized_r_none_when_no_open() -> None:
    """open_unrealized_r is None when no position is still open."""
    from tradinglib.scanner.ledger import _stats

    records = [
        {"status": "target", "r": 2.0, "ticker": "A", "date": "2026-06-01"},
        {"status": "waiting", "r": None, "ticker": "B", "date": "2026-06-01"},
    ]
    stats = _stats(records)
    assert stats["total_r"] == pytest.approx(2.0)
    assert stats["open_unrealized_r"] is None


def test_max_drawdown_r_counts_losses_from_inception() -> None:
    """Peak starts at 0; two consecutive stops of -1 each → max DD == 2.0."""
    from tradinglib.scanner.ledger import _max_drawdown_r

    records = [
        {
            "status": "stopped",
            "r": -1.0,
            "exit_date": "2026-06-01",
            "ticker": "A",
            "date": "2026-05-28",
        },
        {
            "status": "stopped",
            "r": -1.0,
            "exit_date": "2026-06-02",
            "ticker": "B",
            "date": "2026-05-28",
        },
    ]
    assert _max_drawdown_r(records) == pytest.approx(2.0)


def test_combined_stats_unchanged_shape(tmp_path: Path) -> None:
    """The top-level stats block retains existing keys (back-compat)."""
    base = _base_with_watchlist(tmp_path)
    loader = _loader_with({"TST": _tst_bars(), "WLVL": _tst_bars()})
    ledger = build_ledger(base, asof="2026-06-11", loader=loader)
    required_keys = {
        "issued",
        "waiting",
        "expired",
        "open",
        "stopped",
        "target",
        "errors",
        "hit_rate",
        "total_r",
        "avg_r",
    }
    assert required_keys <= set(ledger["stats"])


def test_ledger_respects_row_entry_window(tmp_path: Path) -> None:
    # 6 post-issue sessions, entry never triggered: the default window (5)
    # would expire the row; entry_window=15 keeps it waiting.
    report = {
        "tickets": {"long": [], "short": []},
        "watchlist": {
            "long": [
                {
                    "ticker": "WIN",
                    "stance": "long",
                    "strategy": "setup:pead",
                    "tier": "watch",
                    "entry_window": 15,
                    "levels": {
                        "entry": 200.0,
                        "entry_type": "stop",
                        "stop": 90.0,
                        "target": 420.0,
                    },
                }
            ],
            "short": [],
        },
    }
    (tmp_path / "2026-06-01").mkdir()
    (tmp_path / "2026-06-01" / "report.json").write_text(json.dumps(report), encoding="utf-8")
    idx = pd.date_range("2026-06-02", periods=6, freq="B", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6}, index=idx
    )
    ledger = build_ledger(tmp_path, asof="2026-06-10", loader=lambda t, **kw: bars)
    assert ledger["tickets"][0]["status"] == "waiting"
    assert ledger["tickets"][0]["entry_window"] == 15


# ---------------------------------------------------------------------------
# Closed-ticket freeze + corporate-action flag (A5b)
# ---------------------------------------------------------------------------


def _single_ticket_base(tmp_path: Path, ticket: dict, *, date: str = "2026-06-08") -> Path:
    base = tmp_path / "scans"
    (base / date).mkdir(parents=True)
    (base / date / "report.json").write_text(
        json.dumps(_report(date, [ticket], [])), encoding="utf-8"
    )
    return base


def _tst_stop_bars() -> pd.DataFrame:
    # market entry fills at open 100; same bar low 94 <= stop 95 -> stopped, r=-1.0
    idx = pd.date_range("2026-06-09", periods=2, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 96.0],
            "high": [101.0, 97.0],
            "low": [94.0, 90.0],
            "close": [95.0, 92.0],
        },
        index=idx,
    )


def _tst_trigger_bars() -> pd.DataFrame:
    # for a stop-entry @200: bar0 high 205 triggers fill at 200; bar1 high 211 >= target 210
    idx = pd.date_range("2026-06-09", periods=2, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [201.0, 209.0],
            "high": [205.0, 211.0],
            "low": [200.0, 208.0],
            "close": [203.0, 210.5],
        },
        index=idx,
    )


def test_freeze_closed_ticket_immutable_across_builds(tmp_path: Path) -> None:
    # A closed (target/stopped) ticket's status+R must never change on a later
    # build, even if a split/dividend has silently re-scaled the refreshed bars.
    base = _single_ticket_base(tmp_path, _ticket("TST"))
    l1 = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    write_ledger(l1, base)
    assert l1["tickets"][0]["status"] == "target"
    assert l1["tickets"][0]["r"] == pytest.approx(2.0)

    # rebuild with bars that, if re-simulated, would flip it to stopped/-1.0
    l2 = build_ledger(base, asof="2026-06-20", loader=_loader_with({"TST": _tst_stop_bars()}))
    assert l2["tickets"][0]["status"] == "target"  # frozen, not re-scored
    assert l2["tickets"][0]["r"] == pytest.approx(2.0)


def test_freeze_force_rebuild_re_simulates(tmp_path: Path) -> None:
    # The escape hatch: force_rebuild ignores the prior ledger so a bad early
    # close can be corrected.
    base = _single_ticket_base(tmp_path, _ticket("TST"))
    l1 = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    write_ledger(l1, base)
    l2 = build_ledger(
        base,
        asof="2026-06-20",
        loader=_loader_with({"TST": _tst_stop_bars()}),
        force_rebuild=True,
    )
    assert l2["tickets"][0]["status"] == "stopped"  # re-simulated, not frozen


def test_freeze_does_not_freeze_waiting(tmp_path: Path) -> None:
    # Only target/stopped are frozen; a still-waiting ticket must keep re-simulating
    # as new bars arrive (it can still resolve later).
    base = _single_ticket_base(tmp_path, _ticket("TST", entry=200.0, entry_type="stop"))
    l1 = build_ledger(base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}))
    write_ledger(l1, base)
    assert l1["tickets"][0]["status"] in ("waiting", "expired")
    l2 = build_ledger(base, asof="2026-06-20", loader=_loader_with({"TST": _tst_trigger_bars()}))
    assert l2["tickets"][0]["status"] == "target"  # re-simulated (waiting is not frozen)


def _split_bars() -> pd.DataFrame:
    # A 2:1 split mid-hold: the ADJUSTED OHLC are halved (so a frozen raw-dollar
    # ticket mis-scores against them), while unadj_* carry the raw issue-night tape
    # the levels were actually set on (entry 100, stop 95, target 110).
    idx = pd.date_range("2026-06-09", periods=2, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [50.0, 54.5],
            "high": [50.5, 55.5],
            "low": [49.5, 54.0],
            "close": [50.0, 55.25],
            "unadj_open": [100.0, 109.0],
            "unadj_high": [101.0, 111.0],
            "unadj_low": [99.0, 108.0],
            "unadj_close": [100.0, 110.5],
        },
        index=idx,
    )


def test_ledger_scores_against_unadjusted_when_present(tmp_path: Path) -> None:
    # audit #88: with unadjusted bars available, the ledger scores the frozen dollar
    # levels on the raw tape (target hit, R correct) instead of dropping the ticket —
    # even though a corp action occurred (actions_probe would say True). The A5b drop
    # is the FALLBACK, not the default, once we can score correctly.
    base = _single_ticket_base(tmp_path, _ticket("TST"))
    ledger = build_ledger(
        base,
        asof="2026-06-11",
        loader=_loader_with({"TST": _split_bars()}),
        actions_probe=lambda ticker, start, end: True,
    )
    rec = ledger["tickets"][0]
    assert rec["status"] == "target"
    assert rec["r"] == pytest.approx(2.0)  # (110 - 100) / risk 5, on the raw tape
    assert not rec.get("corp_action_since_issue")  # scored correctly -> not dropped
    assert ledger["stats"]["total_r"] == pytest.approx(2.0)  # kept in realized
    assert ledger["stats"]["corp_action_excluded"] == 0


def test_corp_action_flag_excludes_closed_from_realized(tmp_path: Path) -> None:
    # A closed ticket whose holding window crossed a split/dividend is flagged and
    # dropped from realized stats — the auto_adjust rescale makes its frozen-level
    # scoring unreliable, so we don't trust the win/loss (audit A5b).
    base = _single_ticket_base(tmp_path, _ticket("TST"))
    ledger = build_ledger(
        base,
        asof="2026-06-11",
        loader=_loader_with({"TST": _tst_bars()}),
        actions_probe=lambda ticker, start, end: True,
    )
    rec = ledger["tickets"][0]
    assert rec["status"] == "target"  # still observed
    assert rec["corp_action_since_issue"] is True
    assert ledger["stats"]["total_r"] == pytest.approx(0.0)  # excluded from realized
    assert ledger["stats"]["hit_rate"] is None
    assert ledger["stats"]["corp_action_excluded"] == 1


def test_corp_action_probe_failure_is_non_fatal(tmp_path: Path) -> None:
    # A flaky actions feed must never fail the build; it degrades to flag-absent.
    base = _single_ticket_base(tmp_path, _ticket("TST"))

    def boom(ticker: str, start: str, end: str) -> bool:
        raise RuntimeError("yfinance 429")

    ledger = build_ledger(
        base, asof="2026-06-11", loader=_loader_with({"TST": _tst_bars()}), actions_probe=boom
    )
    rec = ledger["tickets"][0]
    assert rec["status"] == "target"
    assert not rec.get("corp_action_since_issue")
    assert ledger["stats"]["total_r"] == pytest.approx(2.0)  # counts normally


def test_corp_action_flag_survives_freeze(tmp_path: Path) -> None:
    # flag-then-freeze: a flagged close is frozen WITH its flag and stays excluded
    # on later builds even when no probe runs.
    base = _single_ticket_base(tmp_path, _ticket("TST"))
    l1 = build_ledger(
        base,
        asof="2026-06-11",
        loader=_loader_with({"TST": _tst_bars()}),
        actions_probe=lambda ticker, start, end: True,
    )
    write_ledger(l1, base)
    assert l1["tickets"][0]["corp_action_since_issue"] is True

    l2 = build_ledger(base, asof="2026-06-20", loader=_loader_with({"TST": _tst_stop_bars()}))
    rec = l2["tickets"][0]
    assert rec["status"] == "target"  # frozen
    assert rec["corp_action_since_issue"] is True  # flag carried forward
    assert l2["stats"]["total_r"] == pytest.approx(0.0)  # still excluded


def test_stats_excludes_corp_action_flagged_from_realized() -> None:
    from tradinglib.scanner.ledger import _stats

    records = [
        {"status": "target", "r": 2.0, "ticker": "A", "date": "2026-06-01"},
        {
            "status": "stopped",
            "r": -1.0,
            "ticker": "B",
            "date": "2026-06-01",
            "corp_action_since_issue": True,
        },
    ]
    stats = _stats(records)
    assert stats["total_r"] == pytest.approx(2.0)  # B excluded
    assert stats["avg_r"] == pytest.approx(2.0)  # over the one unflagged closed trade
    assert stats["hit_rate"] == pytest.approx(1.0)  # 1 win / 1 realized closed
    assert stats["corp_action_excluded"] == 1
