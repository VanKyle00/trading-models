"""Tests for the S&P 500 constituent-universe loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_FAKE_WIKI_HTML = """
<html><body>
<table id="constituents">
  <thead>
    <tr>
      <th>Symbol</th><th>Security</th><th>GICS Sector</th>
      <th>GICS Sub-Industry</th><th>Headquarters Location</th>
      <th>Date added</th><th>CIK</th><th>Founded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td>
      <td>Technology Hardware</td><td>Cupertino</td>
      <td>1982-11-30</td><td>0000320193</td><td>1977</td>
    </tr>
    <tr>
      <td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td>
      <td>Multi-Sector Holdings</td><td>Omaha</td>
      <td>2010-02-16</td><td>0001067983</td><td>1839</td>
    </tr>
    <tr>
      <td>JNJ</td><td>Johnson &amp; Johnson</td><td>Health Care</td>
      <td>Pharmaceuticals</td><td>New Brunswick</td>
      <td>1973-06-30</td><td>0000200406</td><td>1886</td>
    </tr>
  </tbody>
</table>
</body></html>
"""


class _FakeResponse:
    text = _FAKE_WIKI_HTML

    def raise_for_status(self) -> None:
        pass


def test_canonicalize_schema_and_ticker_normalization() -> None:
    from tradinglib.loaders.universe import sp500 as loader

    raw = pd.read_html(__import__("io").StringIO(_FAKE_WIKI_HTML))[0]
    out = loader._canonicalize(raw)

    assert list(out.columns) == ["ticker", "name", "sector", "sub_industry", "cik"]
    # yfinance uses dashes where Wikipedia uses dots (BRK.B -> BRK-B)
    assert "BRK-B" in set(out["ticker"])
    assert out["cik"].dtype == "int64"
    assert set(out["sector"]) == {"Information Technology", "Financials", "Health Care"}


def test_get_sp500_constituents_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.universe import sp500 as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    calls = {"n": 0}

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse()

    with patch.object(loader.httpx, "get", fake_get):
        first = loader.get_sp500_constituents()
        second = loader.get_sp500_constituents()

    cached = list((tmp_path / "universe" / "sp500").glob("*.parquet"))
    assert len(cached) == 1
    assert calls["n"] == 1  # second read served from the snapshot cache
    assert first.equals(second)
    assert len(first) == 3


def test_get_sp500_constituents_falls_back_to_static_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.universe import sp500 as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    def boom(url: str, **kwargs: object) -> _FakeResponse:
        raise ConnectionError("no network")

    with patch.object(loader.httpx, "get", boom):
        out = loader.get_sp500_constituents()

    assert list(out.columns) == ["ticker", "name", "sector", "sub_industry", "cik"]
    assert len(out) > 400  # the vendored static list is a full S&P 500 snapshot
    # fallback results are not cached, so the next run retries the scrape
    assert not list((tmp_path / "universe" / "sp500").glob("*.parquet"))
