"""Throttled HTTP client for SEC EDGAR.

The SEC's fair-access policy requires a declared User-Agent with a contact
address and caps clients at 10 requests/second; we run at half that. 429/503
responses are retried with exponential backoff. The clock and sleep are
injectable so tests can run the throttle deterministically.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

_USER_AGENT = "trading-models research (van.kyle.00@gmail.com)"
_DEFAULT_RPS = 5.0
_RETRY_STATUSES = (429, 503)


class EdgarClient:
    def __init__(
        self,
        *,
        requests_per_sec: float = _DEFAULT_RPS,
        max_retries: int = 3,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = 1.0 / requests_per_sec
        self._max_retries = max_retries
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self._client = httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get(self, url: str) -> httpx.Response:
        """GET ``url`` with throttling and backoff; raises on HTTP errors."""
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            response = self._client.get(url)
            if response.status_code in _RETRY_STATUSES and attempt < self._max_retries:
                self._sleep(float(2**attempt))
                continue
            break
        assert response is not None
        response.raise_for_status()
        return response
