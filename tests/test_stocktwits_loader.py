"""Tests for the Stocktwits symbol-stream loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "sentiment" / "stocktwits.json").read_text(
        encoding="utf-8"
    )
)


class _Resp:
    def __init__(self, payload: dict | None = None, fail: bool = False) -> None:
        self._payload = payload or {}
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 404")

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.social import stocktwits as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_stocktwits_schema_and_tags(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    def _get(url: str, **kwargs):
        seen.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    df = loader.get_stocktwits("NVDA")
    assert list(df.columns) == ["ticker", "created", "body", "sentiment", "username", "url"]
    assert len(df) == 3
    assert list(df["sentiment"]) == ["Bullish", "Bearish", None]
    assert df.iloc[0]["url"] == "https://stocktwits.com/bull_guy/message/101"
    assert str(df["created"].dt.tz) == "UTC"
    assert "streams/symbol/NVDA.json" in seen[0]


def test_stocktwits_unknown_symbol_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp(fail=True)))
    df = loader.get_stocktwits("ZZZZZZ")
    assert df.empty
    assert list(df.columns) == ["ticker", "created", "body", "sentiment", "username", "url"]


def test_stocktwits_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def _get(url: str, **kwargs):
        calls.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    loader.get_stocktwits("NVDA")
    loader.get_stocktwits("NVDA")
    assert len(calls) == 1
