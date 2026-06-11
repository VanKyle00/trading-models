"""Next ex-dividend date (yfinance calendar).

Uncached by design: the only consumer is the interactive options-planner chat,
which is single-ticker and always wants fresh data (its other loaders run with
``refresh=True`` for the same reason — see tradinglib/assistant/planner.py).
yfinance is mocked in tests and never called live (repo convention).
``Ticker.calendar`` is a dict in yfinance >= 0.2.28 with an 'Ex-Dividend Date'
entry that keeps holding the LAST ex-div date after it passes, so past dates
are filtered to None here.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_next_ex_dividend(ticker: str) -> pd.Timestamp | None:
    """Upcoming ex-dividend date for ``ticker``, or None if none scheduled.

    Returns a naive date-resolution Timestamp (the calendar has no time
    component). Raises on fetch errors — callers isolate, like earnings.
    """
    cal = yf.Ticker(ticker).calendar
    if not isinstance(cal, dict):
        return None  # older yfinance returned a DataFrame; no support, no crash
    raw = cal.get("Ex-Dividend Date")
    if raw is None:
        return None
    ex_div = pd.Timestamp(raw)
    if ex_div.tzinfo is not None:
        ex_div = ex_div.tz_convert("UTC").tz_localize(None)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return ex_div if ex_div >= today else None
