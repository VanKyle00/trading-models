"""Tests for archived-family resolution in the backfill replay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_scan.py"
_spec = importlib.util.spec_from_file_location("backfill_scan", _SCRIPT)
backfill_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill_scan)


def _write_report(base: Path, date: str, long_tickers: list[str], short_tickers: list[str]) -> None:
    (base / date).mkdir(parents=True)
    (base / date / "report.json").write_text(
        json.dumps(
            {
                "fa_candidates": {
                    "long": [{"ticker": t} for t in long_tickers],
                    "short": [{"ticker": t} for t in short_tickers],
                }
            }
        ),
        encoding="utf-8",
    )


def test_archived_family_resolves_newest_report_at_or_before_night(tmp_path: Path) -> None:
    _write_report(tmp_path, "2026-06-01", ["AAA"], ["XXX"])
    _write_report(tmp_path, "2026-06-08", ["BBB"], ["YYY"])
    families = backfill_scan.load_archived_families(tmp_path)

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-06-09"))
    assert fam["long"] == ["BBB"] and source == "2026-06-08"
    assert fam["short"] == ["YYY"]

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-06-08"))
    assert fam["long"] == ["BBB"] and source == "2026-06-08"  # night == report date

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-06-03"))
    assert fam["long"] == ["AAA"] and source == "2026-06-01"

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-05-01"))
    assert fam["long"] == ["AAA"] and source == "fallback"  # before any report
    assert fam["short"] == ["XXX"]


def test_load_archived_families_skips_pre_fa_reports(tmp_path: Path) -> None:
    _write_report(tmp_path, "2026-06-01", ["AAA"], ["XXX"])
    (tmp_path / "2026-06-05").mkdir()
    (tmp_path / "2026-06-05" / "report.json").write_text("{}", encoding="utf-8")
    _write_report(tmp_path, "2026-06-08", ["BBB"], ["YYY"])
    families = backfill_scan.load_archived_families(tmp_path)
    assert [date for date, _ in families] == ["2026-06-01", "2026-06-08"]
