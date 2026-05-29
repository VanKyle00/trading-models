"""Tests for microstructure features (aggregation + order flow imbalance)."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.features.microstructure import (
    aggregate_daily_chunks,
    aggregate_to_bars,
    order_flow_imbalance,
)


def _make_trades(rows: list[tuple[str, float, float, bool]]) -> pd.DataFrame:
    """Build a synthetic trades DataFrame.

    Each row is (timestamp, price, qty, is_buyer_maker).
    """
    ts, prices, qtys, makers = zip(*rows, strict=True)
    idx = pd.DatetimeIndex(pd.to_datetime(list(ts), utc=True), name="timestamp")
    return pd.DataFrame(
        {"price": list(prices), "qty": list(qtys), "is_buyer_maker": list(makers)},
        index=idx,
    )


def test_aggregate_to_bars_produces_ohlcv() -> None:
    # Three trades in the same 60s bucket: low, high, last
    trades = _make_trades(
        [
            ("2024-08-05 00:00:10Z", 100.0, 1.0, False),  # buy
            ("2024-08-05 00:00:20Z", 102.0, 2.0, True),  # sell
            ("2024-08-05 00:00:50Z", 101.0, 0.5, False),  # buy
        ]
    )
    bars = aggregate_to_bars(trades, bar_seconds=60)
    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 102.0
    assert row["low"] == 100.0
    assert row["close"] == 101.0
    assert row["volume"] == pytest.approx(3.5)
    assert row["n_trades"] == 3


def test_ofi_signs() -> None:
    # All aggressive buys → OFI = +1; all aggressive sells → OFI = -1
    buys = _make_trades(
        [
            ("2024-01-01 00:00:01Z", 100.0, 1.0, False),
            ("2024-01-01 00:00:02Z", 100.0, 2.0, False),
        ]
    )
    sells = _make_trades(
        [
            ("2024-01-01 00:00:01Z", 100.0, 1.0, True),
            ("2024-01-01 00:00:02Z", 100.0, 2.0, True),
        ]
    )
    assert order_flow_imbalance(buys, bar_seconds=60).iloc[0] == pytest.approx(1.0)
    assert order_flow_imbalance(sells, bar_seconds=60).iloc[0] == pytest.approx(-1.0)


def test_ofi_balanced_returns_zero() -> None:
    # Equal buy and sell volume → OFI = 0
    trades = _make_trades(
        [
            ("2024-01-01 00:00:01Z", 100.0, 1.0, False),  # buy 1
            ("2024-01-01 00:00:02Z", 100.0, 1.0, True),  # sell 1
        ]
    )
    assert order_flow_imbalance(trades, bar_seconds=60).iloc[0] == pytest.approx(0.0)


def test_buy_and_sell_volume_split() -> None:
    trades = _make_trades(
        [
            ("2024-01-01 00:00:01Z", 100.0, 3.0, False),  # buy 3
            ("2024-01-01 00:00:02Z", 100.0, 1.0, True),  # sell 1
        ]
    )
    bars = aggregate_to_bars(trades, bar_seconds=60)
    row = bars.iloc[0]
    assert row["buy_volume"] == pytest.approx(3.0)
    assert row["sell_volume"] == pytest.approx(1.0)
    assert row["signed_volume"] == pytest.approx(2.0)
    assert row["volume"] == pytest.approx(4.0)
    assert row["ofi"] == pytest.approx(0.5)


def test_multiple_bars_split_correctly() -> None:
    trades = _make_trades(
        [
            ("2024-01-01 00:00:30Z", 100.0, 1.0, False),  # bar ending 00:01
            ("2024-01-01 00:01:30Z", 101.0, 1.0, True),  # bar ending 00:02
            ("2024-01-01 00:02:30Z", 102.0, 1.0, False),  # bar ending 00:03
        ]
    )
    bars = aggregate_to_bars(trades, bar_seconds=60)
    assert len(bars) == 3
    assert bars["close"].tolist() == [100.0, 101.0, 102.0]


def test_aggregate_to_bars_rejects_missing_columns() -> None:
    df = pd.DataFrame(
        {"price": [100.0], "is_buyer_maker": [False]},
        index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
    )
    with pytest.raises(ValueError, match="must have columns"):
        aggregate_to_bars(df, bar_seconds=60)


def test_aggregate_daily_chunks_matches_full_aggregation() -> None:
    # Trades spanning the 08-04 -> 08-05 midnight boundary, including a trade at
    # EXACTLY midnight. With right-closed bars, the bar labelled 00:00:00 (08-05)
    # covers (23:59:00, 00:00:00] and therefore straddles the day boundary: the
    # full aggregation puts the 23:59 trades AND the midnight trade in one bar,
    # while per-day aggregation splits them across two chunks. The streaming
    # aggregator must re-merge that boundary bar to stay identical.
    day1 = _make_trades(
        [
            ("2024-08-04 23:58:30Z", 100.0, 1.0, False),  # bar @ 23:59:00
            ("2024-08-04 23:59:30Z", 101.0, 2.0, True),  # bar @ 00:00:00 (08-05)
            ("2024-08-04 23:59:45Z", 102.0, 1.5, False),  # bar @ 00:00:00 (08-05)
        ]
    )
    day2 = _make_trades(
        [
            ("2024-08-05 00:00:00Z", 103.0, 0.5, True),  # bar @ 00:00:00 (08-05) — straddle
            ("2024-08-05 00:00:30Z", 104.0, 1.0, False),  # bar @ 00:01:00
            ("2024-08-05 00:01:30Z", 105.0, 2.0, True),  # bar @ 00:02:00
        ]
    )
    full = aggregate_to_bars(pd.concat([day1, day2]), bar_seconds=60)
    streamed = aggregate_daily_chunks(iter([day1, day2]), bar_seconds=60)
    pd.testing.assert_frame_equal(streamed, full, check_freq=False)


def test_aggregate_daily_chunks_single_chunk_equals_direct() -> None:
    trades = _make_trades(
        [
            ("2024-08-05 00:00:30Z", 100.0, 1.0, False),  # bar @ 00:01
            ("2024-08-05 00:01:30Z", 101.0, 1.0, True),  # bar @ 00:02
            ("2024-08-05 00:02:30Z", 102.0, 1.0, False),  # bar @ 00:03
        ]
    )
    pd.testing.assert_frame_equal(
        aggregate_daily_chunks(iter([trades]), bar_seconds=60),
        aggregate_to_bars(trades, bar_seconds=60),
        check_freq=False,
    )


def test_aggregate_daily_chunks_consumes_a_generator_lazily() -> None:
    # Passing a generator (not a list) must work, and each chunk should be
    # consumed one at a time so peak memory stays bounded to a single chunk.
    seen: list[int] = []

    def gen():
        for i, day in enumerate(["2024-08-04", "2024-08-05"]):
            seen.append(i)
            yield _make_trades([(f"{day} 00:00:30Z", 100.0 + i, 1.0, False)])

    bars = aggregate_daily_chunks(gen(), bar_seconds=60)
    assert seen == [0, 1]  # generator fully consumed
    assert len(bars) == 2


def test_aggregate_daily_chunks_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no trade chunks"):
        aggregate_daily_chunks(iter([]), bar_seconds=60)


def test_aggregate_daily_chunks_rejects_indivisible_bar_seconds() -> None:
    trades = _make_trades([("2024-08-05 00:00:10Z", 100.0, 1.0, False)])
    with pytest.raises(ValueError, match="divide"):
        aggregate_daily_chunks(iter([trades]), bar_seconds=7)  # 86400 % 7 != 0
