"""Tests for the Russell 1000 constituent-universe loader."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_FAKE_WIKI_HTML = """
<html><body>
<table>
  <thead>
    <tr><th>Company</th><th>Symbol</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
  </thead>
  <tbody>
    <tr><td>Apple Inc.</td><td>AAPL</td><td>Information Technology</td>
        <td>Technology Hardware</td></tr>
    <tr><td>Berkshire Hathaway</td><td>BRK.B</td><td>Financials</td>
        <td>Multi-Sector Holdings</td></tr>
    <tr><td>Newco Without Cik</td><td>NEWCO</td><td>Industrials</td><td>Widgets</td></tr>
  </tbody>
</table>
</body></html>
"""

_CIK_MAP = {"AAPL": 320193, "BRK-B": 1067983}


class _FakeResponse:
    text = _FAKE_WIKI_HTML

    def raise_for_status(self) -> None:
        pass


def _patch_cik_map(monkeypatch: pytest.MonkeyPatch, loader) -> None:
    monkeypatch.setattr(loader, "get_cik_map", lambda *, refresh=False, client=None: _CIK_MAP)


def test_canonicalize_schema_cik_join_and_normalization() -> None:
    from tradinglib.loaders.universe import russell1000 as loader

    raw = pd.read_html(StringIO(_FAKE_WIKI_HTML))[0]
    out = loader._canonicalize(raw, _CIK_MAP)

    assert list(out.columns) == ["ticker", "name", "sector", "sub_industry", "cik"]
    assert out["cik"].dtype == "Int64"  # nullable: not every ticker maps
    indexed = out.set_index("ticker")
    assert indexed.loc["BRK-B", "cik"] == 1067983  # BRK.B -> BRK-B before the join
    assert pd.isna(indexed.loc["NEWCO", "cik"])  # unmapped ticker kept, cik <NA>
    assert indexed.loc["AAPL", "sector"] == "Information Technology"


def test_get_russell1000_constituents_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.universe import russell1000 as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    _patch_cik_map(monkeypatch, loader)
    calls = {"n": 0}

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse()

    with patch.object(loader.httpx, "get", fake_get):
        first = loader.get_russell1000_constituents()
        second = loader.get_russell1000_constituents()

    assert calls["n"] == 1  # second read served from the snapshot cache
    assert first.equals(second)
    assert len(first) == 3
    assert second.attrs["snapshot"] == first.attrs["snapshot"]  # fresh + cache-hit stamped


def test_get_russell1000_constituents_falls_back_to_latest_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.universe import russell1000 as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    _patch_cik_map(monkeypatch, loader)

    # seed an older cached snapshot, then make the scrape fail
    cache_dir = tmp_path / "universe" / "russell1000"
    cache_dir.mkdir(parents=True)
    seeded = pd.DataFrame(
        {
            "ticker": ["OLD"],
            "name": ["Old Snapshot Co"],
            "sector": ["Tech"],
            "sub_industry": ["Software"],
            "cik": pd.array([1], dtype="Int64"),
        }
    )
    seeded.to_parquet(cache_dir / "2026-06-01.parquet")

    def boom(url: str, **kwargs: object) -> _FakeResponse:
        raise ConnectionError("no network")

    with patch.object(loader.httpx, "get", boom):
        out = loader.get_russell1000_constituents()

    assert list(out["ticker"]) == ["OLD"]  # served the stale snapshot, not an error
    assert out.attrs["snapshot"] == "2026-06-01"  # staleness visible to the pipeline


def test_get_russell1000_constituents_raises_without_any_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.universe import russell1000 as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    _patch_cik_map(monkeypatch, loader)

    def boom(url: str, **kwargs: object) -> _FakeResponse:
        raise ConnectionError("no network")

    with patch.object(loader.httpx, "get", boom), pytest.raises(ConnectionError):
        loader.get_russell1000_constituents()


def test_canonicalize_drops_duplicate_tickers() -> None:
    # a noisy Wikipedia edit repeating a symbol must not double-weight a name
    from tradinglib.loaders.universe import russell1000 as loader

    raw = pd.read_html(StringIO(_FAKE_WIKI_HTML))[0]
    doubled = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    out = loader._canonicalize(doubled, _CIK_MAP)

    assert list(out["ticker"]) == sorted(out["ticker"])
    assert out["ticker"].is_unique
    assert len(out) == 3
