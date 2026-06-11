"""Tests for the Bluesky cashtag-search loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "sentiment" / "bluesky.json").read_text(encoding="utf-8")
)


class _Resp:
    def __init__(self, payload: dict | None = None, fail: bool = False) -> None:
        self._payload = payload or {}
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 403")

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.social import bluesky as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_bluesky_schema_fields_and_params(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _get(url: str, **kwargs):
        seen["url"] = url
        seen["params"] = kwargs.get("params") or {}
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    df = loader.get_bluesky_posts("NVDA")
    assert list(df.columns) == ["ticker", "created", "text", "handle", "likes", "reposts", "url"]
    assert len(df) == 3
    assert df.iloc[0]["handle"] == "chipbull.bsky.social"  # API ("top") order preserved
    assert df.iloc[0]["likes"] == 412 and df.iloc[0]["reposts"] == 88
    assert df.iloc[0]["url"] == "https://bsky.app/profile/chipbull.bsky.social/post/3kabc111"
    assert "🚀" in df.iloc[0]["text"]  # UTF-8 survives end to end
    assert df.iloc[2]["url"] == ""  # no author handle -> no link
    assert str(df["created"].dt.tz) == "UTC"
    assert "app.bsky.feed.searchPosts" in seen["url"]
    assert seen["params"]["q"] == "$NVDA"
    assert seen["params"]["sort"] == "top"
    assert seen["params"]["lang"] == "en"
    assert "since" in seen["params"] and seen["params"]["limit"] == 25


def test_bluesky_http_error_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp(fail=True)))
    df = loader.get_bluesky_posts("NVDA")
    assert df.empty
    assert list(df.columns) == ["ticker", "created", "text", "handle", "likes", "reposts", "url"]


def test_bluesky_snapshot_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def _get(url: str, **kwargs):
        calls.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    first = loader.get_bluesky_posts("NVDA")
    second = loader.get_bluesky_posts("NVDA")
    assert len(calls) == 1
    assert first.equals(second)
    assert "🚀" in second.iloc[0]["text"]  # emoji survives the parquet round-trip


def test_bluesky_caps_items(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp(_FIXTURE)))
    df = loader.get_bluesky_posts("NVDA", max_items=1)
    assert len(df) == 1
