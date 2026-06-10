"""planner: bar/chain/earnings plumbing for the options-planner chat tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tradinglib.assistant.planner as planner
from tradinglib.features.technical import atr


def _bars(n: int = 300) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(n) / 5.0)
    idx = pd.date_range("2025-06-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=idx,
    )


def test_propose_levels_long_uses_atr_stop_and_2r_target(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)

    out = planner.propose_levels("test", "long")

    entry = float(bars["close"].iloc[-1])
    atr14 = float(atr(bars["high"], bars["low"], bars["close"], 14).iloc[-1])
    assert out["ticker"] == "TEST"  # upper-cased
    assert out["levels"]["entry"] == pytest.approx(entry, abs=0.01)
    assert out["levels"]["entry_type"] == "market"
    assert out["levels"]["stop"] == pytest.approx(entry - 2 * atr14, abs=0.01)
    assert out["levels"]["target"] == pytest.approx(
        entry + 2 * (entry - out["levels"]["stop"]), abs=0.02
    )
    assert out["atr14"] == pytest.approx(atr14, abs=0.01)
    assert out["asof"] == bars.index[-1].strftime("%Y-%m-%d")


def test_propose_levels_short_mirrors(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)

    out = planner.propose_levels("TEST", "short")

    lv = out["levels"]
    assert lv["stop"] > lv["entry"] > lv["target"]


def test_propose_levels_empty_bars_raises(monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError, match="no daily bars"):
        planner.propose_levels("NOPE", "long")
