"""Tests for the event-driven backtest engine.

Two big things we want to verify:

1. **Equivalence:** A trivial event-driven strategy produces the same
   result as the vectorized engine fed the same signal. The event engine
   should be a thin adapter, not a different math model.
2. **Stateful logic works:** Strategies with internal state (e.g., a
   trailing stop) behave as expected and respect the one-bar lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest import (
    Bar,
    EventEngine,
    bars_from_dataframe,
    run_backtest,
    run_event_backtest,
)


@pytest.fixture
def rising_df() -> pd.DataFrame:
    """50 bars of 1%/bar geometric uptrend."""
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    prices = 100.0 * (1.01 ** np.arange(50))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.full(50, 1000.0),
        },
        index=idx,
    )


def test_bars_from_dataframe_roundtrip(rising_df: pd.DataFrame) -> None:
    bars = bars_from_dataframe(rising_df)
    assert len(bars) == len(rising_df)
    assert bars[0].close == pytest.approx(rising_df["close"].iloc[0])
    assert bars[-1].close == pytest.approx(rising_df["close"].iloc[-1])
    assert isinstance(bars[0].timestamp, pd.Timestamp)


def test_bars_from_dataframe_missing_columns() -> None:
    df = pd.DataFrame({"open": [1.0], "close": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        bars_from_dataframe(df)


def test_always_long_matches_vectorized(rising_df: pd.DataFrame) -> None:
    """An 'always long' callback must produce identical PnL to the vectorized
    engine fed a constant +1 signal."""

    class AlwaysLong:
        def on_bar(self, engine: EventEngine, bar: Bar) -> None:
            engine.set_target_position(1.0)

    bars = bars_from_dataframe(rising_df)
    event_result = run_event_backtest(bars, AlwaysLong(), fee_bps=0, slippage_bps=0)

    constant_signal = pd.Series(1.0, index=rising_df.index)
    vec_result = run_backtest(rising_df["close"], constant_signal, fee_bps=0, slippage_bps=0)

    pd.testing.assert_series_equal(
        event_result.equity_curve,
        vec_result.equity_curve,
        check_names=False,
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        event_result.returns,
        vec_result.returns,
        check_names=False,
        check_freq=False,
    )


def test_flat_strategy_zero_equity_growth(rising_df: pd.DataFrame) -> None:
    class Flat:
        def on_bar(self, engine: EventEngine, bar: Bar) -> None:
            pass  # never sets a target → default 0.0

    bars = bars_from_dataframe(rising_df)
    result = run_event_backtest(bars, Flat(), fee_bps=0, slippage_bps=0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_signal_lag_prevents_lookahead(rising_df: pd.DataFrame) -> None:
    """A strategy that 'cheats' by looking at the current bar's close should
    still not be able to capture *this* bar's return — the one-bar lag
    enforced by the underlying vectorized engine should hold."""

    class GreedyOmniscient:
        def on_bar(self, engine: EventEngine, bar: Bar) -> None:
            engine.set_target_position(1.0)  # always long going forward

    bars = bars_from_dataframe(rising_df)
    result = run_event_backtest(bars, GreedyOmniscient(), fee_bps=0, slippage_bps=0)
    # Position at bar 0 must be 0 (the lag); it only ramps up from bar 1.
    assert result.position.iloc[0] == 0.0
    assert result.position.iloc[1] == 1.0


def test_trailing_stop_exits_position() -> None:
    """A trailing-stop strategy must exit when price drops by the stop
    threshold from the in-trade peak."""

    class TrailingStopAt5pct:
        def __init__(self) -> None:
            self.peak: float | None = None

        def on_bar(self, engine: EventEngine, bar: Bar) -> None:
            if engine.position_fraction == 0.0:
                engine.set_target_position(1.0)  # enter immediately
                self.peak = bar.close
                return
            assert self.peak is not None
            self.peak = max(self.peak, bar.close)
            if bar.close < self.peak * 0.95:  # 5% trailing stop
                engine.set_target_position(0.0)

    # Path: ramp up to 110, then drop to 100 — a 9% drop from peak triggers the stop
    idx = pd.date_range("2024-01-01", periods=15, freq="D")
    closes = np.concatenate([np.linspace(100, 110, 11), np.linspace(108, 100, 4)])
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000.0,
        },
        index=idx,
    )

    bars = bars_from_dataframe(df)
    result = run_event_backtest(bars, TrailingStopAt5pct(), fee_bps=0, slippage_bps=0)
    # The trailing stop should have flipped position to 0 by the end
    assert result.position.iloc[-1] == 0.0
    # And we should have *captured* some of the uptrend before exiting
    assert result.equity_curve.iloc[-1] > 100_000.0


def test_empty_bars_raises() -> None:
    class Anything:
        def on_bar(self, engine: EventEngine, bar: Bar) -> None: ...

    with pytest.raises(ValueError, match="no bars"):
        run_event_backtest(iter([]), Anything())


def test_event_engine_target_setter() -> None:
    engine = EventEngine()
    assert engine.position_fraction == 0.0
    engine.set_target_position(0.5)
    assert engine.position_fraction == 0.5
    engine.set_target_position(-1.0)
    assert engine.position_fraction == -1.0
