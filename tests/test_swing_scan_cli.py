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
        "funnel": {
            "universe": 5,
            "fa_shortlist": 5,
            "with_setups": 1,
            "tournament_candidates": 2,
            "tournament_survivors": 1,
            "tickets": 1,
        },
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
        "fa_candidates": {
            "long": [
                {"ticker": "AAPL", "name": "Apple", "sector": "Tech", "fa_score": 0.9, "rank": 1}
            ],
            "short": [],
        },
        "tournament": {
            "long": [
                {
                    "ticker": "AAPL",
                    "winner": {
                        "strategy": "sma_cross",
                        "levels": {
                            "entry": 210.0,
                            "stop": 195.0,
                            "target": 225.0,
                        },
                    },
                    "winner_changed": False,
                }
            ],
            "short": [],
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
                        "target": 225.0,
                        "condition": "enter on the cross",
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
                    "quotes_asof": "2026-06-09",
                    "iv_ratio": None,
                    "next_earnings": None,
                    "account_size": 100000.0,
                    "risk_per_trade_pct": 0.01,
                    "structures": [
                        {
                            "kind": "stock",
                            "label": "buy 100 shares",
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
                            "quantity": 4,
                            "recommended": True,
                            "loss_at_stop": None,
                        }
                    ],
                    "warnings": ["indicative quotes: last/close marks"],
                }
            ],
            "short": [],
        },
    }


def test_cli_writes_report_and_passes_config(
    swing_scan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seen: dict = {}

    def fake_run_scan(config, provider=None, previous_report=None, recent_ledger=None):
        seen["config"] = config
        seen["provider"] = provider
        return _canned_result()

    monkeypatch.setattr(swing_scan, "run_scan", fake_run_scan)

    exit_code = swing_scan.main(
        [
            "--limit",
            "5",
            "--fa-keep",
            "10",
            "--top",
            "3",
            "--skip-llm",
            "--universe",
            "sp500",
            "--short-keep",
            "5",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen["config"].limit == 5
    assert seen["config"].fa_keep == 10
    assert seen["config"].top == 3
    assert seen["config"].skip_llm is True
    assert seen["config"].universe == "sp500"
    assert seen["config"].short_keep == 5
    assert seen["provider"] is None

    out_dir = tmp_path / "2026-06-09"
    loaded = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert loaded["candidates"][0]["ticker"] == "AAPL"
    assert (out_dir / "report.md").exists()

    printed = capsys.readouterr().out
    assert "AAPL" in printed


def test_cli_passes_sizing_flags(
    swing_scan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def fake_run_scan(config, provider=None, previous_report=None, recent_ledger=None):
        seen["config"] = config
        return _canned_result()

    monkeypatch.setattr(swing_scan, "run_scan", fake_run_scan)

    swing_scan.main(
        [
            "--skip-llm",
            "--account-size",
            "250000",
            "--risk-per-trade-pct",
            "0.005",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert seen["config"].account_size == 250000.0
    assert seen["config"].risk_per_trade_pct == 0.005


def test_cli_builds_claude_provider_unless_skipped(
    swing_scan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tradinglib.assistant.provider as provider_module

    class _StubClaude:
        pass

    monkeypatch.setattr(provider_module, "ClaudeProvider", _StubClaude)

    seen: dict = {}

    def fake_run_scan(config, provider=None, previous_report=None, recent_ledger=None):
        seen["provider"] = provider
        return _canned_result()

    monkeypatch.setattr(swing_scan, "run_scan", fake_run_scan)

    swing_scan.main(["--out-dir", str(tmp_path)])

    assert isinstance(seen["provider"], _StubClaude)
