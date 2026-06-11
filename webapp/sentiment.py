"""View helper for the /sentiment page — ticker validation + engine call."""

from __future__ import annotations

import re

_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


def valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_RE.match(ticker.strip()))


def get_report(ticker: str, *, refresh: bool = False) -> dict:
    from tradinglib.sentiment.report import run_sentiment  # lazy: keeps webapp import light

    return run_sentiment(ticker, refresh=refresh).to_dict()
