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


def _write_report(base: Path, date: str, long_tickers: list[str]) -> None:
    (base / date).mkdir(parents=True)
    (base / date / "report.json").write_text(
        json.dumps(
            {
                "fa_candidates": {
                    "long": [{"ticker": t} for t in long_tickers],
                    "short": [{"ticker": "SHRT"}],
                }
            }
        ),
        encoding="utf-8",
    )


def test_archived_family_resolves_newest_report_at_or_before_night(tmp_path: Path) -> None:
    _write_report(tmp_path, "2026-06-01", ["AAA"])
    _write_report(tmp_path, "2026-06-08", ["BBB"])
    families = backfill_scan.load_archived_families(tmp_path)

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-06-09"))
    assert fam["long"] == ["BBB"] and source == "2026-06-08"

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-06-03"))
    assert fam["long"] == ["AAA"] and source == "2026-06-01"

    fam, source = backfill_scan.family_for_night(families, pd.Timestamp("2026-05-01"))
    assert fam["long"] == ["AAA"] and source == "fallback"  # before any report
