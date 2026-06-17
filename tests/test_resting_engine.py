"""Tests for the intrabar resting-order scoring engine (tournament A2).

The load-bearing guarantee is cert==ticket: run_resting_backtest fills via the
SAME primitives as simulate_ticket, so a resting strategy's tournament score
describes the trade the forward ledger actually scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.backtest.resting_engine import run_resting_backtest, simulate_resting
from tradinglib.strategist.evaluate import simulate_ticket


def _bars(opens, highs, lows, closes) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(opens), freq="B", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def _orders(bars: pd.DataFrame, *, entry, stop, target, entry_type, armed=True) -> pd.DataFrame:
    """A constant resting order resting on every bar (caller is responsible for shifting)."""
    return pd.DataFrame(
        {
            "entry": float(entry),
            "stop": float(stop),
            "target": float(target),
            "entry_type": entry_type,
            "armed": armed,
        },
        index=bars.index,
    )


def test_fill_identity_with_simulate_ticket() -> None:
    # A resting buy-stop at 105; bar 1 triggers (high 106), bar 2 hits target 115.
    bars = _bars(
        opens=[100.0, 104.0, 110.0],
        highs=[101.0, 106.0, 116.0],
        lows=[99.0, 103.0, 109.0],
        closes=[100.0, 105.5, 115.0],
    )
    orders = _orders(bars, entry=105.0, stop=100.0, target=115.0, entry_type="stop")
    trades = simulate_resting(bars, orders, stance="long")
    assert trades, "expected a fill"
    t0 = trades[0]

    ticket = {
        "stance": "long",
        "levels": {"entry": 105.0, "entry_type": "stop", "stop": 100.0, "target": 115.0},
    }
    # asof before all bars -> simulate_ticket sees the same bars; big window so it does not expire
    sim = simulate_ticket(ticket, bars, asof="2024-01-01", entry_window=10)

    assert t0["entry_fill"] == sim["entry_fill"]
    assert t0["exit_fill"] == sim["exit_fill"]
    assert t0["status"] == sim["status"]


def test_both_touched_is_stopped_and_ambiguous() -> None:
    # entry bar 0 fills the stop-entry at 100 (open); same bar touches both stop 95 and target 110
    bars = _bars(opens=[100.0], highs=[111.0], lows=[94.0], closes=[100.0])
    orders = _orders(bars, entry=100.0, stop=95.0, target=110.0, entry_type="stop")
    trades = simulate_resting(bars, orders, stance="long")
    assert len(trades) == 1
    assert trades[0]["status"] == "stopped"  # worst case wins
    assert trades[0]["ambiguous"] is True


def test_limit_entry_bar_target_suppressed() -> None:
    # limit buy at 100 fills bar 0 (low 99); target 110 also touched bar 0 but is SUPPRESSED
    # on the entry bar; bar 1 then hits the target.
    bars = _bars(
        opens=[101.0, 109.0],
        highs=[111.0, 112.0],
        lows=[99.0, 108.0],
        closes=[105.0, 110.5],
    )
    orders = _orders(bars, entry=100.0, stop=90.0, target=110.0, entry_type="limit")
    trades = simulate_resting(bars, orders, stance="long")
    assert len(trades) == 1
    assert trades[0]["entry_idx"] == 0
    assert trades[0]["status"] == "target"
    assert trades[0]["exit_idx"] == 1  # not the entry bar


def test_single_bar_round_trip_counts_and_attributes() -> None:
    # stop-entry at 100 fills bar 0 (open 100); same bar low 94 hits stop 95 -> one-bar trip.
    bars = _bars(opens=[100.0], highs=[101.0], lows=[94.0], closes=[96.0])
    orders = _orders(bars, entry=100.0, stop=95.0, target=120.0, entry_type="stop")
    result = run_resting_backtest(bars, orders, stance="long")
    assert result.n_completed_trades == 1
    # the single bar earns entry_fill(100) -> exit_fill(95): -5% on a long
    assert result.returns.iloc[0] < 0.0
    assert float(result.position.iloc[0]) == 1.0


def test_re_arm_counts_each_completed_trade() -> None:
    # two non-overlapping stop-entry round trips -> exactly 2 completed trades
    bars = _bars(
        opens=[100.0, 96.0, 100.0, 96.0],
        highs=[101.0, 97.0, 101.0, 97.0],
        lows=[94.0, 90.0, 94.0, 90.0],
        closes=[96.0, 95.0, 96.0, 95.0],
    )
    orders = _orders(bars, entry=100.0, stop=95.0, target=120.0, entry_type="stop")
    result = run_resting_backtest(bars, orders, stance="long")
    assert result.n_completed_trades == 2


def test_open_trade_at_end_is_not_counted_but_marks_to_close() -> None:
    # enters bar 0, never exits (no stop/target touch) -> open at slice end
    bars = _bars(
        opens=[100.0, 101.0, 102.0],
        highs=[101.0, 102.0, 103.0],
        lows=[100.0, 101.0, 102.0],
        closes=[101.0, 102.0, 103.0],
    )
    orders = _orders(bars, entry=100.0, stop=90.0, target=200.0, entry_type="stop")
    result = run_resting_backtest(bars, orders, stance="long")
    assert result.n_completed_trades == 0  # the open trade is not a completed round-trip
    assert float(result.position.iloc[-1]) == 1.0  # still held at the end


def test_backtest_result_contract() -> None:
    bars = _bars(
        opens=[100.0, 104.0, 110.0],
        highs=[101.0, 106.0, 116.0],
        lows=[99.0, 103.0, 109.0],
        closes=[100.0, 105.5, 115.0],
    )
    orders = _orders(bars, entry=105.0, stop=100.0, target=115.0, entry_type="stop")
    result = run_resting_backtest(bars, orders, stance="long", initial_capital=100_000.0)
    assert result.returns.index.equals(bars.index)
    assert result.equity_curve.iloc[-1] == (1.0 + result.returns).prod() * 100_000.0
    assert {"sharpe", "deflated_sharpe"} <= set(result.metrics)
    assert np.isfinite(result.returns).all()
