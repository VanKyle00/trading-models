"""Tests for the yfinance news loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_NESTED_ITEM = {
    "id": "abc",
    "content": {
        "title": "Apple unveils new product",
        "summary": "A big launch event.",
        "pubDate": "2026-06-05T12:00:00Z",
        "provider": {"displayName": "Reuters"},
    },
}
_FLAT_ITEM = {
    "title": "Apple beats estimates",
    "publisher": "Bloomberg",
    "providerPublishTime": 1780610400,  # 2026-06-04 ~14:00 UTC
    "summary": "Strong quarter.",
}


def _fake_ticker(items: list[dict]) -> type:
    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def news(self) -> list[dict]:
            return items

    return _FakeTicker


def test_news_canonical_schema_handles_both_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.news import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "Ticker", _fake_ticker([_NESTED_ITEM, _FLAT_ITEM])):
        df = loader.get_news("AAPL")

    assert list(df.columns) == ["ticker", "published", "title", "summary", "publisher"]
    assert set(df["publisher"]) == {"Reuters", "Bloomberg"}
    assert str(df["published"].dt.tz) == "UTC"


def test_news_caps_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.news import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    many = [_NESTED_ITEM] * 40
    with patch.object(loader.yf, "Ticker", _fake_ticker(many)):
        df = loader.get_news("AAPL", max_items=15)

    assert len(df) == 15


def test_news_empty_is_fine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.news import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "Ticker", _fake_ticker([])):
        df = loader.get_news("ZZZZ")

    assert df.empty
    assert list(df.columns) == ["ticker", "published", "title", "summary", "publisher"]


def test_news_snapshot_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.news import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    calls = {"n": 0}

    class _CountingTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def news(self) -> list[dict]:
            calls["n"] += 1
            return [_NESTED_ITEM]

    with patch.object(loader.yf, "Ticker", _CountingTicker):
        first = loader.get_news("AAPL")
        second = loader.get_news("AAPL")

    assert calls["n"] == 1
    assert first.equals(second)
    cached = list((tmp_path / "news" / "yfinance" / "AAPL").glob("*.parquet"))
    assert len(cached) == 1


def test_news_dedupes_pandas_friendly_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.news import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "Ticker", _fake_ticker([_FLAT_ITEM])):
        df = loader.get_news("AAPL")

    assert df.iloc[0]["published"] == pd.Timestamp(1780610400, unit="s", tz="UTC")
