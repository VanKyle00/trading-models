"""Event-driven backtest engine.

Processes bars one at a time and dispatches each to a user-supplied
:class:`Strategy` callback. The strategy reads the bar, optionally calls
``engine.set_target_position(...)`` to update its target, and the engine
records the resulting signal series. After all bars are processed, the
recorded signal is fed through the vectorized engine so PnL math and
metrics are bit-identical to a vectorized backtest.

The benefit over the vectorized engine is *imperative expressiveness*:
stop-losses, trailing stops, regime filters, "only flip after N
consecutive bars", and other stateful per-bar logic are much cleaner to
write as a sequence of callbacks than as vector operations. The
:class:`BacktestResult` shape is identical to what
:func:`tradinglib.backtest.run_backtest` returns, so results from either
engine plug into the same metrics / plotting pipeline.

When to use which
-----------------
Vectorized (:func:`run_backtest`):
    Signal is a pure function of price history (SMA crossover, ML
    predictions, z-scores). Faster and simpler.

Event-driven (:func:`run_event_backtest`):
    Path-dependent rules; stateful filters; tracking custom variables
    bar-by-bar. Also the stepping-stone toward tick-level processing for
    microstructure models, which will share this dispatch pattern.

Example
-------
A trailing-stop wrapper around an SMA crossover signal::

    from tradinglib.backtest.event_engine import (
        Bar, Strategy, EventEngine,
        bars_from_dataframe, run_event_backtest,
    )

    class SmaWithTrailingStop:
        def __init__(self, fast=50, slow=200, stop_pct=0.10):
            self.fast = fast; self.slow = slow; self.stop_pct = stop_pct
            self.closes = []
            self.peak = None

        def on_bar(self, engine: EventEngine, bar: Bar) -> None:
            self.closes.append(bar.close)
            if len(self.closes) < self.slow:
                return
            fast_ma = sum(self.closes[-self.fast:]) / self.fast
            slow_ma = sum(self.closes[-self.slow:]) / self.slow

            if engine.position_fraction > 0:
                self.peak = max(self.peak, bar.close)
                if bar.close < self.peak * (1 - self.stop_pct):
                    engine.set_target_position(0.0)   # trailing stop hit
                    self.peak = None
                    return

            if fast_ma > slow_ma and engine.position_fraction == 0:
                engine.set_target_position(1.0)
                self.peak = bar.close
            elif fast_ma < slow_ma and engine.position_fraction > 0:
                engine.set_target_position(0.0)
                self.peak = None

    bars = bars_from_dataframe(daily_ohlcv_dataframe)
    result = run_event_backtest(bars, SmaWithTrailingStop())
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from tradinglib.backtest.engine import BacktestResult, run_backtest


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar — the smallest event the v1 engine consumes."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class Strategy(Protocol):
    """Strategies implement a single per-bar callback."""

    def on_bar(self, engine: EventEngine, bar: Bar) -> None: ...


class EventEngine:
    """Per-bar dispatcher used by :func:`run_event_backtest`.

    The strategy talks to the engine through :meth:`set_target_position`
    and reads :attr:`position_fraction` to know its current target. The
    engine deliberately keeps no PnL state — accounting is delegated to
    the vectorized engine after the event loop completes.
    """

    def __init__(self) -> None:
        self._target: float = 0.0

    def set_target_position(self, fraction: float) -> None:
        """Set the target position as a fraction of equity (-inf, +inf)."""
        self._target = float(fraction)

    @property
    def position_fraction(self) -> float:
        """The strategy's currently-requested target."""
        return self._target


def run_event_backtest(
    bars: Iterable[Bar],
    strategy: Strategy,
    *,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run an event-driven backtest by calling ``strategy.on_bar`` per bar.

    The signal recorded at bar ``t`` is the strategy's target *after*
    seeing bar ``t``'s close. The vectorized engine then lags it one bar
    when computing PnL, so the new position applies starting at bar
    ``t+1`` — same look-ahead-bias guardrail as the vectorized engine.
    """
    engine = EventEngine()
    timestamps: list[pd.Timestamp] = []
    closes: list[float] = []
    signals: list[float] = []

    for bar in bars:
        timestamps.append(bar.timestamp)
        closes.append(bar.close)
        strategy.on_bar(engine, bar)
        signals.append(engine.position_fraction)

    if not timestamps:
        raise ValueError("no bars provided to run_event_backtest")

    idx = pd.DatetimeIndex(timestamps)
    prices = pd.Series(closes, index=idx, name="close")
    signal_series = pd.Series(signals, index=idx, name="signal")

    return run_backtest(
        prices,
        signal_series,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        periods_per_year=periods_per_year,
    )


def bars_from_dataframe(df: pd.DataFrame) -> list[Bar]:
    """Convert a canonical OHLCV DataFrame into a list of :class:`Bar` events.

    Expects columns ``open, high, low, close, volume`` and a
    :class:`~pandas.DatetimeIndex`.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    return [
        Bar(
            timestamp=pd.Timestamp(ts),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for ts, row in df.iterrows()
    ]
