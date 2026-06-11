"""Tests for the /sentiment page and API route (engine stubbed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp.main import create_app


def test_sentiment_page_renders() -> None:
    resp = TestClient(create_app()).get("/sentiment")
    assert resp.status_code == 200
    assert "sentiment" in resp.text.lower()


def test_sentiment_api_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake(ticker: str, *, refresh: bool = False) -> dict:
        captured["ticker"] = ticker
        captured["refresh"] = refresh
        return {"ticker": ticker.upper(), "status": "ok", "tiers": []}

    monkeypatch.setattr("webapp.sentiment.get_report", _fake)
    resp = TestClient(create_app()).get("/api/v1/sentiment/nvda?refresh=1")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "NVDA"
    assert captured == {"ticker": "nvda", "refresh": True}


def test_sentiment_api_invalid_ticker_400() -> None:
    resp = TestClient(create_app()).get("/api/v1/sentiment/NOT%20A%20TICKER!!")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_sentiment_api_engine_failure_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ticker: str, *, refresh: bool = False) -> dict:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("webapp.sentiment.get_report", _boom)
    resp = TestClient(create_app()).get("/api/v1/sentiment/NVDA")
    assert resp.status_code == 500
    assert "error" in resp.json()
