"""Tests for the Binance crypto loader.

Network calls are stubbed — these tests exercise canonicalization,
pagination, caching, and the empty-response error path against a synthetic
HTTP response.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_kline(open_time_ms: int, open_p: float, close_p: float) -> list:
    """Build one synthetic kline row matching Binance's array format."""
    close_time_ms = open_time_ms + 24 * 60 * 60 * 1000 - 1
    return [
        open_time_ms,
        str(open_p),  # open
        str(open_p * 1.02),  # high
        str(open_p * 0.98),  # low
        str(close_p),  # close
        "1000.5",  # volume
        close_time_ms,
        "1234567.89",  # quote volume
        100,  # n_trades
        "500.25",  # taker_buy_base
        "617283.94",  # taker_buy_quote
        "0",  # ignore
    ]


@pytest.fixture
def fake_klines_one_batch() -> list:
    """5 daily klines, fewer than MAX_LIMIT → pagination terminates after one call."""
    start = 1_704_067_200_000  # 2024-01-01 UTC
    day_ms = 24 * 60 * 60 * 1000
    return [_make_kline(start + i * day_ms, 100.0 + i, 101.0 + i) for i in range(5)]


def _make_response(payload: list) -> MagicMock:
    resp = MagicMock()
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp


def test_canonicalize_produces_expected_schema(fake_klines_one_batch: list) -> None:
    from tradinglib.loaders.crypto.binance import _canonicalize

    out = _canonicalize(fake_klines_one_batch, "BTCUSDT")
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "symbol"]
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "timestamp"
    assert (out["symbol"] == "BTCUSDT").all()
    assert out["open"].dtype == "float64"
    assert out["volume"].dtype == "float64"
    assert len(out) == 5


def test_load_daily_single_batch(
    fake_klines_one_batch: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.crypto import binance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response(fake_klines_one_batch))

    with patch.object(loader.httpx, "Client", return_value=client):
        df = loader.load_daily("BTCUSDT")

    cached = tmp_path / "binance" / "BTCUSDT" / "daily.parquet"
    assert cached.exists()
    assert len(df) == 5
    # Only one HTTP call needed since the batch was below MAX_LIMIT
    assert client.get.call_count == 1


def test_load_daily_paginates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.crypto import binance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    # Shrink MAX_LIMIT so we can fake pagination without 1000 rows
    monkeypatch.setattr(loader, "MAX_LIMIT", 3)

    start_ms = 1_704_067_200_000
    day_ms = 24 * 60 * 60 * 1000
    batch1 = [_make_kline(start_ms + i * day_ms, 100.0 + i, 101.0 + i) for i in range(3)]
    batch2 = [_make_kline(start_ms + (3 + i) * day_ms, 110.0 + i, 111.0 + i) for i in range(2)]

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(side_effect=[_make_response(batch1), _make_response(batch2)])

    with patch.object(loader.httpx, "Client", return_value=client):
        df = loader.load_daily("BTCUSDT")

    assert len(df) == 5  # batches stitched together
    assert client.get.call_count == 2


def test_load_daily_uses_cache(
    fake_klines_one_batch: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.crypto import binance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response(fake_klines_one_batch))

    with patch.object(loader.httpx, "Client", return_value=client) as patched_cls:
        loader.load_daily("BTCUSDT")
        loader.load_daily("BTCUSDT")
        # Second call must read parquet, not construct a new HTTP client
        assert patched_cls.call_count == 1


def test_load_daily_filters_by_date(
    fake_klines_one_batch: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.crypto import binance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response(fake_klines_one_batch))

    with patch.object(loader.httpx, "Client", return_value=client):
        df = loader.load_daily("BTCUSDT", start="2024-01-02", end="2024-01-04")

    assert len(df) == 3
    assert df.index.min() == pd.Timestamp("2024-01-02", tz="UTC")
    assert df.index.max() == pd.Timestamp("2024-01-04", tz="UTC")


def test_load_daily_raises_on_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.crypto import binance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response([]))

    with (
        patch.object(loader.httpx, "Client", return_value=client),
        pytest.raises(ValueError, match="no data"),
    ):
        loader.load_daily("ZZZZUSDT")


def test_to_ms_handles_string_and_datetime() -> None:
    from datetime import datetime

    from tradinglib.loaders.crypto.binance import _to_ms

    assert _to_ms("2024-01-01") == 1_704_067_200_000
    assert _to_ms(datetime(2024, 1, 1, tzinfo=UTC)) == 1_704_067_200_000
