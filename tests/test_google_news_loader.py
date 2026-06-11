"""Tests for the Google News RSS loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = (Path(__file__).parent / "fixtures" / "sentiment" / "google_news.xml").read_text(
    encoding="utf-8"
)


class _Resp:
    def __init__(self, text: str = "", fail: bool = False) -> None:
        self.text = text
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 503")


def _fake_httpx(text: str = _FIXTURE, *, fail: bool = False, calls: list | None = None):
    def _get(url: str, **kwargs):
        if calls is not None:
            calls.append(url)
        return _Resp(text, fail=fail)

    return SimpleNamespace(get=_get)


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.news import google_news as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_google_news_schema_and_order(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(loader, "httpx", _fake_httpx(calls=calls))
    df = loader.get_google_news("NVDA")
    assert list(df.columns) == ["ticker", "published", "title", "publisher", "url"]
    assert len(df) == 2
    assert df.iloc[0]["publisher"] == "Reuters"  # newest first
    assert df.iloc[0]["url"] == "https://news.example.com/nvda-rally"
    assert str(df["published"].dt.tz) == "UTC"
    assert "rss/search" in calls[0] and "NVDA+stock" in calls[0]


def test_google_news_http_error_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", _fake_httpx(fail=True))
    df = loader.get_google_news("NVDA")
    assert df.empty
    assert list(df.columns) == ["ticker", "published", "title", "publisher", "url"]


def test_google_news_snapshot_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(loader, "httpx", _fake_httpx(calls=calls))
    first = loader.get_google_news("NVDA")
    second = loader.get_google_news("NVDA")
    assert len(calls) == 1
    assert first.equals(second)


def test_google_news_caps_items(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", _fake_httpx())
    df = loader.get_google_news("NVDA", max_items=1)
    assert len(df) == 1
