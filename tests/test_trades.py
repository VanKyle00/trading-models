# tests/test_trades.py
import pandas as pd

from tradinglib.eval.trades import trades_from_position


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_single_long_round_trip():
    idx = _idx(5)
    position = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=idx)
    prices = pd.Series([10.0, 10.0, 11.0, 12.0, 12.0], index=idx)
    trades = trades_from_position(position, prices)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["side"] == "long"
    assert row["entry_time"] == idx[1]
    assert row["exit_time"] == idx[3]
    assert row["entry_price"] == 10.0
    assert row["exit_price"] == 12.0
    assert row["pnl"] == 2.0          # (12 - 10) * +1 size
    assert row["duration"] == 2       # bars held


def test_open_position_at_end_is_closed_on_last_bar():
    idx = _idx(3)
    position = pd.Series([0.0, 1.0, 1.0], index=idx)
    prices = pd.Series([10.0, 10.0, 13.0], index=idx)
    trades = trades_from_position(position, prices)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_time"] == idx[2]
    assert trades.iloc[0]["pnl"] == 3.0


def test_short_trade_pnl_sign():
    idx = _idx(4)
    position = pd.Series([0.0, -1.0, -1.0, 0.0], index=idx)
    prices = pd.Series([10.0, 10.0, 9.0, 8.0], index=idx)
    trades = trades_from_position(position, prices)
    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "short"
    assert trades.iloc[0]["pnl"] == 2.0   # (8 - 10) * -1 size


def test_flat_throughout_returns_empty():
    idx = _idx(4)
    position = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    prices = pd.Series([10.0, 11.0, 12.0, 13.0], index=idx)
    trades = trades_from_position(position, prices)
    assert trades.empty
    assert list(trades.columns) == [
        "entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl", "duration"
    ]
