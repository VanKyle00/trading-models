# tests/test_trades.py
import pandas as pd
import pytest

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
    assert row["pnl"] == 2.0  # (12 - 10) * +1 size
    assert row["duration"] == 2  # bars held


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
    assert trades.iloc[0]["pnl"] == 2.0  # (8 - 10) * -1 size


def test_flat_throughout_returns_empty():
    idx = _idx(4)
    position = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    prices = pd.Series([10.0, 11.0, 12.0, 13.0], index=idx)
    trades = trades_from_position(position, prices)
    assert trades.empty
    assert list(trades.columns) == [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "pnl",
        "duration",
    ]


def test_long_to_short_flip_produces_two_trades():
    idx = _idx(5)
    position = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0], index=idx)
    prices = pd.Series([10.0, 10.0, 11.0, 11.0, 9.0], index=idx)
    trades = trades_from_position(position, prices)
    assert len(trades) == 2
    assert trades.iloc[0]["side"] == "long"
    assert trades.iloc[0]["pnl"] == 1.0  # (11 - 10) * +1
    assert trades.iloc[1]["side"] == "short"
    assert trades.iloc[1]["pnl"] == 2.0  # (9 - 11) * -1


def test_include_open_false_excludes_dangling_position():
    # An open position at the series end is force-closed by default (display
    # behavior), but include_open=False drops that phantom round-trip so it
    # cannot pad a trade count (audit A3).
    idx = _idx(3)
    position = pd.Series([0.0, 1.0, 1.0], index=idx)  # open at the end
    prices = pd.Series([10.0, 10.0, 13.0], index=idx)
    assert len(trades_from_position(position, prices)) == 1  # default unchanged
    assert trades_from_position(position, prices, include_open=False).empty


def test_include_open_false_keeps_closed_trades():
    # A closed round-trip followed by a dangling open: include_open=False keeps
    # only the closed one.
    idx = _idx(6)
    position = pd.Series([0.0, 1.0, 0.0, 0.0, 1.0, 1.0], index=idx)
    prices = pd.Series([10.0, 10.0, 12.0, 12.0, 12.0, 14.0], index=idx)
    closed = trades_from_position(position, prices, include_open=False)
    assert len(closed) == 1
    assert closed.iloc[0]["exit_time"] == idx[2]


def test_missing_price_raises():
    idx = _idx(3)
    position = pd.Series([0.0, 1.0, 0.0], index=idx)
    prices = pd.Series([10.0, 11.0], index=idx[:2])  # no price for idx[2]
    with pytest.raises(ValueError, match="missing values"):
        trades_from_position(position, prices)
