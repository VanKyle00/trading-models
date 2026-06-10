"""Tests for the SEC ticker -> CIK mapping loader."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tradinglib.loaders.edgar_client import EdgarClient

_PAYLOAD = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "3": {"cik_str": 1067983, "ticker": "BRK-A", "title": "BERKSHIRE HATHAWAY INC"},
}


def _client(calls: dict) -> EdgarClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_PAYLOAD)

    return EdgarClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_get_cik_map_parses_sec_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.universe import cik_map as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    calls = {"n": 0}

    mapping = loader.get_cik_map(client=_client(calls))

    assert mapping["AAPL"] == 320193
    assert mapping["BRK-B"] == 1067983  # SEC already uses dash-style share classes
    assert len(mapping) == 4


def test_get_cik_map_caches_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.universe import cik_map as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    calls = {"n": 0}
    client = _client(calls)

    first = loader.get_cik_map(client=client)
    second = loader.get_cik_map(client=client)

    assert calls["n"] == 1  # second read served from the snapshot cache
    assert first == second
    assert len(list((tmp_path / "universe" / "cik_map").glob("*.parquet"))) == 1


def test_get_cik_map_rejects_empty_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # an empty 200 must raise (and not poison the day's cache with an empty map)
    from tradinglib.loaders.universe import cik_map as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = EdgarClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)

    with pytest.raises(ValueError, match="empty"):
        loader.get_cik_map(client=client)
    assert not list((tmp_path / "universe" / "cik_map").glob("*.parquet"))
