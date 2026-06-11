"""Tests for the yfinance ex-dividend loader."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def _ticker_with(calendar):
    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.calendar = calendar

    return _FakeTicker


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def test_future_ex_dividend_is_returned() -> None:
    from tradinglib.loaders.events import dividends as loader

    ex_div = (_today() + pd.Timedelta(days=5)).date()
    with patch.object(loader.yf, "Ticker", _ticker_with({"Ex-Dividend Date": ex_div})):
        out = loader.get_next_ex_dividend("AAPL")

    assert out == pd.Timestamp(ex_div)


def test_past_ex_dividend_is_none() -> None:
    # yfinance keeps the LAST ex-div in the calendar after it passes
    from tradinglib.loaders.events import dividends as loader

    ex_div = (_today() - pd.Timedelta(days=30)).date()
    with patch.object(loader.yf, "Ticker", _ticker_with({"Ex-Dividend Date": ex_div})):
        assert loader.get_next_ex_dividend("AAPL") is None


def test_today_counts_as_upcoming() -> None:
    from tradinglib.loaders.events import dividends as loader

    with patch.object(loader.yf, "Ticker", _ticker_with({"Ex-Dividend Date": _today().date()})):
        assert loader.get_next_ex_dividend("AAPL") == _today()


def test_missing_entry_is_none() -> None:
    from tradinglib.loaders.events import dividends as loader

    with patch.object(loader.yf, "Ticker", _ticker_with({"Earnings Date": "2026-08-20"})):
        assert loader.get_next_ex_dividend("GOOG") is None


def test_non_dict_calendar_is_none() -> None:
    # older yfinance returned a DataFrame here
    from tradinglib.loaders.events import dividends as loader

    with patch.object(loader.yf, "Ticker", _ticker_with(pd.DataFrame())):
        assert loader.get_next_ex_dividend("AAPL") is None
