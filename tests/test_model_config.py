"""Tests for the per-model ticker configuration helpers."""

from __future__ import annotations

import pytest

from app.model_config import default_ticker, ticker_choices, ticker_mode


def test_free_text_mode() -> None:
    m = {"name": "Classical", "tickers": "any", "default_ticker": "SPY"}
    assert ticker_mode(m) == "free"
    assert default_ticker(m) == "SPY"
    assert ticker_choices(m) == []


def test_fixed_single() -> None:
    m = {"name": "Micro", "tickers": ["BTCUSDT"]}
    assert ticker_mode(m) == "fixed"
    assert ticker_choices(m) == ["BTCUSDT"]
    assert default_ticker(m) == "BTCUSDT"


def test_choice_list() -> None:
    m = {"name": "Multi", "tickers": ["SPY", "QQQ", "AAPL"]}
    assert ticker_mode(m) == "choice"
    assert ticker_choices(m) == ["SPY", "QQQ", "AAPL"]
    assert default_ticker(m) == "SPY"


def test_missing_default_raises() -> None:
    with pytest.raises(ValueError):
        default_ticker({"name": "Bad", "tickers": "any"})


def test_unspecified_tickers_defaults_to_fixed_and_has_no_default() -> None:
    m = {"name": "NoTickers"}
    assert ticker_mode(m) == "fixed"
    assert ticker_choices(m) == []
    with pytest.raises(ValueError):
        default_ticker(m)
