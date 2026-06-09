"""Tests for the DoltHub historical option-chain loader (httpx mocked, never live)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_FAKE_ROWS = [
    {
        "date": "2026-06-05",
        "act_symbol": "AAPL",
        "expiration": "2026-06-18",
        "strike": "290.00",
        "call_put": "Call",
        "bid": "7.10",
        "ask": "7.30",
        "vol": "0.2935",
    },
    {
        "date": "2026-06-05",
        "act_symbol": "AAPL",
        "expiration": "2026-06-18",
        "strike": "290.00",
        "call_put": "Put",
        "bid": "6.80",
        "ask": "7.00",
        "vol": "0.2712",
    },
]

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv"]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _ok(rows: list[dict]) -> _FakeResponse:
    return _FakeResponse({"query_execution_status": "Success", "rows": rows})


def test_canonicalize_schema_and_coercion() -> None:
    from tradinglib.loaders.options import dolthub

    out = dolthub._canonicalize(_FAKE_ROWS, "AAPL")

    assert list(out.columns) == _COLUMNS
    assert set(out["right"]) == {"call", "put"}
    assert out["strike"].dtype == "float64"
    assert out["bid"].iloc[0] == pytest.approx(7.10) or out["bid"].iloc[0] == pytest.approx(6.80)
    assert pd.api.types.is_datetime64_any_dtype(out["expiration"])
    assert out["expiration"].dt.tz is None  # tz-naive by contract


def test_canonicalize_empty_returns_canonical_empty() -> None:
    from tradinglib.loaders.options import dolthub

    out = dolthub._canonicalize([], "AAPL")
    assert list(out.columns) == _COLUMNS
    assert out.empty


def test_load_chain_caches_and_skips_network_on_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.options import dolthub

    monkeypatch.setattr(dolthub, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(dolthub.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        assert "date" in params["q"] and "act_symbol" in params["q"]  # PK-prefix discipline
        return _ok(_FAKE_ROWS)

    monkeypatch.setattr(dolthub.httpx, "get", fake_get)

    df1 = dolthub.load_chain("AAPL", "2026-06-05")
    df2 = dolthub.load_chain("AAPL", "2026-06-05")

    assert calls["n"] == 1  # second call served from parquet cache
    assert len(df1) == 2 and len(df2) == 2
    cached = list((tmp_path / "options" / "dolthub" / "AAPL").glob("*.parquet"))
    assert len(cached) == 1


def test_load_chain_empty_result_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import dolthub

    monkeypatch.setattr(dolthub, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(dolthub.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        return _ok([])

    monkeypatch.setattr(dolthub.httpx, "get", fake_get)

    df1 = dolthub.load_chain("ZZZZ", "2026-06-05")
    df2 = dolthub.load_chain("ZZZZ", "2026-06-05")

    assert df1.empty and df2.empty
    assert calls["n"] == 1  # the miss is cached too


def test_query_error_status_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import dolthub

    monkeypatch.setattr(dolthub, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(dolthub.time, "sleep", lambda s: None)

    def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
        return _FakeResponse(
            {
                "query_execution_status": "Error",
                "query_execution_message": "context deadline exceeded",
            }
        )

    monkeypatch.setattr(dolthub.httpx, "get", fake_get)

    with pytest.raises(RuntimeError, match="context deadline exceeded"):
        dolthub.load_chain("AAPL", "2026-06-06")
