"""Smoke test for the evaluate_tickets CLI (offline: empty report dir).

scripts/ is not a package; load the module by path like test_swing_scan_cli.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tickets.py"


@pytest.fixture
def evaluate_tickets():
    spec = importlib.util.spec_from_file_location("evaluate_tickets", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_tickets"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("evaluate_tickets", None)


def test_cli_builds_and_writes_ledger(
    evaluate_tickets, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    base = tmp_path / "scans"
    (base / "2026-06-10").mkdir(parents=True)
    (base / "2026-06-10" / "report.json").write_text(
        json.dumps({"asof": "2026-06-10", "tickets": {"long": [], "short": []}}),
        encoding="utf-8",
    )

    assert evaluate_tickets.main(["--base", str(base)]) == 0

    ledger = json.loads((base / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["stats"]["issued"] == 0
    out = capsys.readouterr().out
    assert "0 tickets" in out
    assert "ledger.json" in out
