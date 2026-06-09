"""Tests for the swing-scan CLI entry point (pipeline mocked)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "swing_scan.py"


@pytest.fixture
def swing_scan():
    spec = importlib.util.spec_from_file_location("swing_scan", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["swing_scan"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["swing_scan"]


def _canned_result() -> dict:
    return {
        "asof": "2026-06-09",
        "config": {"fa_keep": 40, "top": 15},
        "funnel": {"universe": 5, "fa_shortlist": 5, "with_setups": 1},
        "candidates": [
            {
                "ticker": "AAPL",
                "name": "Apple",
                "sector": "Tech",
                "fa_score": 0.9,
                "setup_score": 0.7,
                "final_score": 0.79,
                "pinned": False,
                "pinned_reason": "",
                "setups": [
                    {
                        "setup_type": "base_breakout",
                        "score": 0.7,
                        "trigger_level": 210.0,
                        "stop_level": 195.0,
                        "evidence": {},
                    }
                ],
                "earnings_warning": False,
            }
        ],
        "errors": [],
    }


def test_cli_writes_report_and_passes_config(
    swing_scan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seen: dict = {}

    def fake_run_scan(config, provider=None):
        seen["config"] = config
        seen["provider"] = provider
        return _canned_result()

    monkeypatch.setattr(swing_scan, "run_scan", fake_run_scan)

    exit_code = swing_scan.main(
        ["--limit", "5", "--fa-keep", "10", "--top", "3", "--skip-llm", "--out-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert seen["config"].limit == 5
    assert seen["config"].fa_keep == 10
    assert seen["config"].top == 3
    assert seen["config"].skip_llm is True
    assert seen["provider"] is None

    out_dir = tmp_path / "2026-06-09"
    loaded = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert loaded["candidates"][0]["ticker"] == "AAPL"
    assert (out_dir / "report.md").exists()

    printed = capsys.readouterr().out
    assert "AAPL" in printed
