"""Tests for the nightly regime overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.scanner.regime import STRATEGY_STYLES, gate_reason, regime_state


def _close(values: np.ndarray) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_regime_state_uptrend() -> None:
    state = regime_state(_close(np.linspace(80.0, 120.0, 300)))
    assert state["trend"] == "up"
    assert state["close"] > state["sma200"]
    assert 0.0 <= state["vol_pctile"] <= 1.0


def test_regime_state_downtrend() -> None:
    state = regime_state(_close(np.linspace(120.0, 80.0, 300)))
    assert state["trend"] == "down"


def test_regime_state_insufficient_history_is_neutral() -> None:
    state = regime_state(_close(np.linspace(100.0, 101.0, 50)))
    assert state["trend"] == "neutral"  # never gates


def test_gate_reason_blocks_meanrev_against_trend() -> None:
    down = {"trend": "down"}
    up = {"trend": "up"}
    assert gate_reason("rsi2", "long", down) == "meanrev long in down-trend regime"
    assert gate_reason("rsi2", "long", up) is None
    assert gate_reason("rsi2", "short", up) == "meanrev short in up-trend regime"
    assert gate_reason("sma_cross", "long", down) is None  # trend style ungated
    assert gate_reason("setup:pead", "long", down) is None  # event style ungated
    assert gate_reason("rsi2", "long", {"trend": "neutral"}) is None


def test_every_registered_strategy_has_a_style() -> None:
    from tradinglib.tournament.strategies import STRATEGIES

    assert set(STRATEGIES) <= set(STRATEGY_STYLES)
    # Pin the gate's input taxonomy: a future "sync" with StrategyDef.style must fail loudly.
    assert STRATEGY_STYLES["rsi2"] == "meanrev"
    assert STRATEGY_STYLES["bollinger"] == "meanrev"
    assert STRATEGY_STYLES["ma_pullback"] == "trend"  # deliberate divergence from registry style
    assert STRATEGY_STYLES["pead"] == "event"


def test_regime_state_drops_nan_closes() -> None:
    values = np.linspace(80.0, 120.0, 300)
    values[250] = np.nan  # a stray NaN inside the SMA window must not poison it into gating "down"
    state = regime_state(_close(values))
    assert state["trend"] == "up"


def test_regime_state_nan_heavy_series_is_neutral() -> None:
    values = np.linspace(80.0, 120.0, 300)
    values[100:] = np.nan  # only 100 valid closes remain
    state = regime_state(_close(values))
    assert state["trend"] == "neutral"
