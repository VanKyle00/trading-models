"""Tests for the Levels contract and its stop/target helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.tournament.levels import Levels, direction, protective_stop, two_r_target


def _flat_bars(n: int = 60, close: float = 100.0, spread: float = 1.0) -> pd.DataFrame:
    c = np.full(n, close)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread, "close": c}, index=idx)


def test_direction_maps_stances() -> None:
    assert direction("long") == 1
    assert direction("short") == -1
    with pytest.raises(ValueError, match="stance"):
        direction("sideways")


def test_protective_stop_is_two_atr_each_side() -> None:
    bars = _flat_bars()  # true range == 2.0 every bar -> ATR(14) == 2.0
    assert protective_stop(bars, 100.0, "long") == 96.0
    assert protective_stop(bars, 100.0, "short") == 104.0


def test_protective_stop_rejects_degenerate_atr() -> None:
    bars = _flat_bars(spread=0.0)  # high == low == close -> ATR == 0
    with pytest.raises(ValueError, match="ATR"):
        protective_stop(bars, 100.0, "long")


def test_two_r_target_signed() -> None:
    assert two_r_target(100.0, 96.0) == 108.0  # long: R=4 above
    assert two_r_target(100.0, 104.0) == 92.0  # short: R=4 below


def test_levels_as_dict_round_trips() -> None:
    lv = Levels(entry=100.0, entry_type="stop", stop=96.0, target=108.0, condition="x")
    assert lv.as_dict() == {
        "entry": 100.0,
        "entry_type": "stop",
        "stop": 96.0,
        "target": 108.0,
        "condition": "x",
    }
