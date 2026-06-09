"""Discovery + loading of swing-scan reports for the /scans pages.

Reports are written by ``scripts/swing_scan.py`` (or ``run_scan`` directly)
to ``data/processed/scans/<YYYY-MM-DD>/report.json``. Date strings are
validated strictly before touching the filesystem.
"""

from __future__ import annotations

import json
import re

from tradinglib.data.paths import processed_dir

SOURCE = "scans"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def list_scan_dates() -> list[str]:
    """Scan dates with a report.json, newest first."""
    base = processed_dir(SOURCE)
    if not base.exists():
        return []
    dates = [
        p.name
        for p in base.iterdir()
        if p.is_dir() and _DATE_RE.match(p.name) and (p / "report.json").exists()
    ]
    return sorted(dates, reverse=True)


def load_scan(date: str) -> dict | None:
    """The parsed report for one date, or None if absent/invalid."""
    if not _DATE_RE.match(date):
        return None
    path = processed_dir(SOURCE) / date / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
