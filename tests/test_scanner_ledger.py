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
