"""Tests for the replay's machine-enforced point-in-time / leak tagging (audit B1).

The replay's selection-layer survivorship (today-vintage FA family, current
membership) was previously only a docstring/JSON caveat. These tests pin the
per-record machine-enforced flag so absolute-R reads cannot be silently
mistaken for a point-in-time-honest backtest.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_scan.py"
_spec = importlib.util.spec_from_file_location("backfill_scan", _SCRIPT)
backfill_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill_scan)


def test_frozen_family_is_leaked() -> None:
    # The frozen family is today's FA report back-projected onto every night:
    # always a selection leak.
    assert backfill_scan.point_in_time_status("frozen", None) == ("frozen", True)
    assert backfill_scan.point_in_time_status("frozen", "anything") == ("frozen", True)


def test_archived_family_is_point_in_time() -> None:
    # A night served by an archived report dated <= night is point-in-time honest.
    assert backfill_scan.point_in_time_status("archived", "2026-06-01") == ("archived", False)


def test_archived_fallback_is_leaked() -> None:
    # Nights before archived coverage reuse the oldest report = future info.
    assert backfill_scan.point_in_time_status("archived", "fallback") == ("fallback", True)


def test_honesty_summary_flags_any_leak() -> None:
    leaked = [{"leak": True}, {"leak": False}]
    summary = backfill_scan.honesty_summary(leaked)
    assert summary["leaked"] is True
    assert summary["leaked_records"] == 1
    assert summary["honest_records"] == 1

    clean = [{"leak": False}, {"leak": False}]
    summary = backfill_scan.honesty_summary(clean)
    assert summary["leaked"] is False
    assert summary["leaked_records"] == 0
