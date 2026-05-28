"""Tests for technical feature primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.features.technical import (
    log_return,
    price_to_sma_ratio,
    realized_volatility,
    rsi,
)


@pytest.fixture
def flat_prices() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.Series(100.0, index=idx)


@pytest.fixture
def rising_prices() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.Series(100.0 * (1.01 ** np.arange(50)), index=idx)


def test_log_return_zero_for_flat(flat_prices: pd.Series) -> None:
    out = log_return(flat_prices, 1)
    assert (out.dropna() == 0.0).all()


def test_log_return_constant_for_geometric(rising_prices: pd.Series) -> None:
    out = log_return(rising_prices, 1).dropna()
    expected = np.log(1.01)
    assert np.allclose(out, expected)


def test_realized_volatility_zero_for_flat(flat_prices: pd.Series) -> None:
    rets = log_return(flat_prices, 1)
    assert (realized_volatility(rets, 5).dropna() == 0.0).all()


def test_rsi_bounded(rising_prices: pd.Series) -> None:
    r = rsi(rising_prices, 14).dropna()
    assert (r >= 0).all()
    assert (r <= 100).all()


def test_rsi_max_on_pure_uptrend(rising_prices: pd.Series) -> None:
    # A series with only positive deltas has no losses, so RSI saturates at 100.
    assert rsi(rising_prices, 14).dropna().iloc[-1] == pytest.approx(100.0)


def test_price_to_sma_ratio_zero_for_flat(flat_prices: pd.Series) -> None:
    assert (price_to_sma_ratio(flat_prices, 5).dropna() == 0.0).all()


def test_price_to_sma_positive_for_uptrend(rising_prices: pd.Series) -> None:
    # In an uptrend, current price is always above its trailing SMA.
    out = price_to_sma_ratio(rising_prices, 5).dropna()
    assert (out > 0).all()
