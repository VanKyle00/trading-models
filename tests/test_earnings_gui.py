"""run_for_gui surfaces a helpful note when the earnings calendar is empty.

The model's default ticker is SPY — an ETF with no earnings — so the default
workbench run trades nothing and returns flat equity. Without a note that reads
as a broken model; the note makes the empty result intentional and actionable.
"""

from __future__ import annotations

import importlib.util

import pandas as pd

from tradinglib.data.paths import repo_root


def _load_model_module():
    model_dir = repo_root() / "models/options/03-earnings-straddle-spy"
    spec = importlib.util.spec_from_file_location("_earnings_straddle", model_dir / "backtest.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calendar(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"earnings_datetime": pd.to_datetime(dates, utc=True)})


def test_run_for_gui_emits_note_when_no_earnings(monkeypatch) -> None:
    module = _load_model_module()
    monkeypatch.setattr(module, "get_earnings_dates", lambda *a, **k: _calendar([]))

    out = module.run_for_gui("2023-01-01", "2024-12-31", symbol="SPY")

    assert out["report"] is None
    assert out["note"]
    assert "SPY" in out["note"]


def test_run_for_gui_has_no_note_when_earnings_present(monkeypatch) -> None:
    module = _load_model_module()
    quarterly = ["2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01", "2024-05-01"]
    monkeypatch.setattr(module, "get_earnings_dates", lambda *a, **k: _calendar(quarterly))

    out = module.run_for_gui("2023-01-01", "2024-12-31", symbol="SPY")

    assert out["report"] is not None
    assert out["note"] is None
