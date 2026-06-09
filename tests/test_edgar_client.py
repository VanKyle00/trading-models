"""Tests for the throttled SEC EDGAR HTTP client."""

from __future__ import annotations

import httpx
import pytest

from tradinglib.loaders.edgar_client import _USER_AGENT, EdgarClient


class _FakeTime:
    """Deterministic clock: time only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _client(handler, fake: _FakeTime, **kwargs) -> EdgarClient:
    return EdgarClient(
        transport=httpx.MockTransport(handler),
        clock=fake.clock,
        sleep=fake.sleep,
        **kwargs,
    )


def test_declares_contact_user_agent() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, text="ok")

    fake = _FakeTime()
    _client(handler, fake).get("https://data.sec.gov/x")

    assert seen["ua"] == _USER_AGENT
    assert "@" in seen["ua"]  # SEC requires a contact address


def test_throttles_consecutive_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    fake = _FakeTime()
    client = _client(handler, fake, requests_per_sec=5.0)
    client.get("https://data.sec.gov/a")
    client.get("https://data.sec.gov/b")

    # second request had to wait ~1/5s
    assert fake.sleeps
    assert sum(fake.sleeps) == pytest.approx(0.2, abs=0.05)


def test_retries_on_429_with_backoff() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, text="ok")

    fake = _FakeTime()
    response = _client(handler, fake).get("https://data.sec.gov/x")

    assert response.status_code == 200
    assert calls["n"] == 3
    # exponential backoff sleeps are present (1s then 2s) among the throttle sleeps
    assert any(s >= 1.0 for s in fake.sleeps)


def test_gives_up_after_max_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    fake = _FakeTime()
    with pytest.raises(httpx.HTTPStatusError):
        _client(handler, fake, max_retries=2).get("https://data.sec.gov/x")


def test_raises_on_client_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    fake = _FakeTime()
    with pytest.raises(httpx.HTTPStatusError):
        _client(handler, fake).get("https://data.sec.gov/x")
