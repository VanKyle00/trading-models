"""Tests for the Seeking Alpha RSS loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = (Path(__file__).parent / "fixtures" / "sentiment" / "seeking_alpha.xml").read_text(
    encoding="utf-8"
)


class _Resp:
    def __init__(self, text: str = "", fail: bool = False) -> None:
        self.text = text
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 403")


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.forums import seeking_alpha as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_seeking_alpha_schema(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _get(url: str, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers") or {}
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    df = loader.get_seeking_alpha("NVDA")
    assert list(df.columns) == ["ticker", "published", "title", "url"]
    assert len(df) == 2
    assert df.iloc[0]["title"].startswith("Nvidia: The Data Center")
    assert "combined/NVDA.xml" in seen["url"]
    assert "User-Agent" in seen["headers"]


def test_seeking_alpha_blocked_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp("", fail=True))
    )
    df = loader.get_seeking_alpha("NVDA")
    assert df.empty
    assert list(df.columns) == ["ticker", "published", "title", "url"]


def test_seeking_alpha_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def _get(url: str, **kwargs):
        calls.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    loader.get_seeking_alpha("NVDA")
    loader.get_seeking_alpha("NVDA")
    assert len(calls) == 1
