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


def test_events_for_queries_the_requested_window_not_module_constants(monkeypatch) -> None:
    """_events_for must scan the actual price window, not the model's 2023-2024
    constants — otherwise a 2022 run (e.g. NVDA) reports 'no earnings' even though
    the name had earnings in that window.
    """
    module = _load_model_module()
    captured: dict[str, object] = {}

    def fake_get_earnings_dates(tickers, start=None, end=None, **kwargs):
        captured["start"] = start
        captured["end"] = end
        return _calendar([])

    monkeypatch.setattr(module, "get_earnings_dates", fake_get_earnings_dates)
    close = pd.Series(
        range(20), index=pd.date_range("2022-01-03", periods=20, freq="B"), name="close"
    )

    module._events_for("NVDA", close)

    assert pd.Timestamp(captured["start"]) == close.index[0]
    assert pd.Timestamp(captured["end"]) == close.index[-1]
