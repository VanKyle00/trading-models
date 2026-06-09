"""Tests for the yfinance fundamentals-snapshot loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_FULL_INFO = {
    "marketCap": 3_000_000_000_000,
    "totalRevenue": 400_000_000_000,
    "revenueGrowth": 0.08,
    "earningsGrowth": 0.12,
    "operatingMargins": 0.30,
    "debtToEquity": 150.0,
    "freeCashflow": 100_000_000_000,
    "forwardPE": 28.0,
    "trailingPE": 32.0,
    "returnOnEquity": 1.5,
    "averageVolume": 50_000_000,
}


def _fake_ticker_factory(info_by_symbol: dict[str, dict | Exception]) -> type:
    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict:
            value = info_by_symbol[self.symbol]
            if isinstance(value, Exception):
                raise value
            return value

    return _FakeTicker


def test_snapshot_maps_info_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.fundamentals import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "Ticker", _fake_ticker_factory({"AAPL": _FULL_INFO})):
        df = loader.get_fundamental_snapshot(["AAPL"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["market_cap"] == 3_000_000_000_000
    assert row["revenue_growth"] == pytest.approx(0.08)
    assert row["operating_margin"] == pytest.approx(0.30)
    assert row["forward_pe"] == pytest.approx(28.0)
    assert row["total_revenue"] == 400_000_000_000


def test_missing_info_keys_become_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.fundamentals import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    sparse = {"marketCap": 1_000_000_000}  # everything else absent
    with patch.object(loader.yf, "Ticker", _fake_ticker_factory({"XYZ": sparse})):
        df = loader.get_fundamental_snapshot(["XYZ"])

    row = df.iloc[0]
    assert row["market_cap"] == 1_000_000_000
    assert np.isnan(row["revenue_growth"])
    assert np.isnan(row["forward_pe"])


def test_one_bad_ticker_does_not_kill_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.fundamentals import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    mapping: dict[str, dict | Exception] = {
        "AAPL": _FULL_INFO,
        "BAD": RuntimeError("yfinance exploded"),
    }
    with patch.object(loader.yf, "Ticker", _fake_ticker_factory(mapping)):
        df = loader.get_fundamental_snapshot(["AAPL", "BAD"])

    assert set(df["ticker"]) == {"AAPL", "BAD"}
    bad = df[df["ticker"] == "BAD"].iloc[0]
    assert np.isnan(bad["market_cap"])
    good = df[df["ticker"] == "AAPL"].iloc[0]
    assert good["market_cap"] == 3_000_000_000_000


def test_snapshot_is_cached_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.fundamentals import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    calls = {"n": 0}

    class _CountingTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict:
            calls["n"] += 1
            return _FULL_INFO

    with patch.object(loader.yf, "Ticker", _CountingTicker):
        first = loader.get_fundamental_snapshot(["AAPL"])
        second = loader.get_fundamental_snapshot(["AAPL"])

    assert calls["n"] == 1
    cached = list((tmp_path / "fundamentals" / "yfinance" / "AAPL").glob("*.parquet"))
    assert len(cached) == 1
    assert first.equals(second)
