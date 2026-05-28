"""Tests for the Binance historical aggTrades loader.

The download path is stubbed — these tests exercise CSV parsing,
canonicalization, caching, and the multi-day stitch path against a
synthetic ZIP built in-memory.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_zip(
    rows: list[list[str]], inner_name: str = "BTCUSDT-aggTrades-2024-08-05.csv"
) -> bytes:
    """Build a Binance-style aggTrades ZIP from a list of CSV rows."""
    csv_text = "\n".join(",".join(row) for row in rows) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, csv_text)
    return buf.getvalue()


@pytest.fixture
def fake_zip() -> bytes:
    """Two trades: one aggressive buy, one aggressive sell."""
    return _make_fake_zip(
        [
            ["1", "58161.00", "0.0026", "100", "100", "1722816000182", "False", "True"],
            ["2", "58160.99", "0.0051", "101", "101", "1722816000353", "True", "True"],
        ]
    )


def _make_response(payload: bytes) -> MagicMock:
    resp = MagicMock()
    resp.content = payload
    resp.raise_for_status = MagicMock()
    return resp


def test_parse_zip_returns_canonical_schema(fake_zip: bytes) -> None:
    from tradinglib.loaders.crypto.binance_trades import _parse_zip

    df = _parse_zip(fake_zip)
    assert list(df.columns) == ["price", "qty", "is_buyer_maker"]
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    assert df["price"].dtype == "float64"
    assert df["qty"].dtype == "float64"
    assert df["is_buyer_maker"].dtype == "bool"
    assert df["is_buyer_maker"].tolist() == [False, True]
    assert len(df) == 2


def test_load_trades_single_day(
    fake_zip: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.crypto import binance_trades as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response(fake_zip))

    with patch.object(loader.httpx, "Client", return_value=client):
        df = loader.load_trades("BTCUSDT", "2024-08-05", "2024-08-05")

    cached = tmp_path / "binance" / "BTCUSDT" / "aggTrades" / "2024-08-05.parquet"
    assert cached.exists()
    assert len(df) == 2
    assert client.get.call_count == 1


def test_load_trades_multi_day_stitches(
    fake_zip: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.crypto import binance_trades as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response(fake_zip))

    with patch.object(loader.httpx, "Client", return_value=client):
        df = loader.load_trades("BTCUSDT", "2024-08-04", "2024-08-06")

    # 3 days x 2 trades = 6 rows
    assert len(df) == 6
    # 3 separate HTTP calls (one per day)
    assert client.get.call_count == 3
    # 3 cached parquets
    cache_root = tmp_path / "binance" / "BTCUSDT" / "aggTrades"
    assert sum(1 for _ in cache_root.glob("*.parquet")) == 3


def test_load_trades_uses_cache(
    fake_zip: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.crypto import binance_trades as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=_make_response(fake_zip))

    with patch.object(loader.httpx, "Client", return_value=client):
        loader.load_trades("BTCUSDT", "2024-08-05", "2024-08-05")
        loader.load_trades("BTCUSDT", "2024-08-05", "2024-08-05")
        # Second call must hit the cache
        assert client.get.call_count == 1


def test_load_trades_rejects_reversed_range(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.crypto import binance_trades as loader

    with pytest.raises(ValueError, match="is after end"):
        loader.load_trades("BTCUSDT", "2024-08-10", "2024-08-05")


def test_to_date_accepts_string_and_date() -> None:
    from tradinglib.loaders.crypto.binance_trades import _to_date

    assert _to_date("2024-08-05") == date(2024, 8, 5)
    assert _to_date(date(2024, 8, 5)) == date(2024, 8, 5)
