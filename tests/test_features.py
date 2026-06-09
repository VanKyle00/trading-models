"""Tests for technical feature primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.features.technical import (
    atr,
    log_return,
    ma_slope,
    pct_off_52w_high,
    price_to_sma_ratio,
    range_tightness,
    realized_volatility,
    relative_strength,
    rsi,
    sma,
    volume_ratio,
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


def test_sma_flat_equals_price(flat_prices: pd.Series) -> None:
    out = sma(flat_prices, 5).dropna()
    assert (out == 100.0).all()


def test_sma_matches_rolling_mean(rising_prices: pd.Series) -> None:
    out = sma(rising_prices, 10)
    expected = rising_prices.rolling(10).mean()
    assert out.equals(expected)


def test_atr_constant_range() -> None:
    # high - low is constantly 2 and the close never gaps, so ATR == 2.
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    high = pd.Series(101.0, index=idx)
    low = pd.Series(99.0, index=idx)
    close = pd.Series(100.0, index=idx)
    out = atr(high, low, close, 14).dropna()
    assert np.allclose(out, 2.0)


def test_atr_includes_gaps() -> None:
    # A gap beyond the bar's own range widens the true range via prev close.
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    high = pd.Series([101.0, 111.0, 111.0], index=idx)
    low = pd.Series([99.0, 109.0, 109.0], index=idx)
    close = pd.Series([100.0, 110.0, 110.0], index=idx)
    out = atr(high, low, close, 2)
    # Bar 1 TR = max(2, |111-100|, |109-100|) = 11; bar 2 TR = 2.
    assert out.iloc[2] == pytest.approx((11.0 + 2.0) / 2.0)


def test_ma_slope_zero_for_flat(flat_prices: pd.Series) -> None:
    out = ma_slope(flat_prices, 5, 5).dropna()
    assert (out == 0.0).all()


def test_ma_slope_positive_for_uptrend(rising_prices: pd.Series) -> None:
    out = ma_slope(rising_prices, 5, 5).dropna()
    assert (out > 0).all()


def test_pct_off_52w_high_zero_at_new_high(rising_prices: pd.Series) -> None:
    # A monotonically rising series is always at its rolling high.
    out = pct_off_52w_high(rising_prices, window=20).dropna()
    assert (out == 0.0).all()


def test_pct_off_52w_high_negative_after_drop() -> None:
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    prices = pd.Series([100.0] * 20 + [80.0] * 10, index=idx)
    out = pct_off_52w_high(prices, window=20)
    assert out.iloc[-1] == pytest.approx(-0.2)


def test_range_tightness_smaller_for_tighter_range() -> None:
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(100.0, index=idx)
    tight = range_tightness(pd.Series(101.0, index=idx), pd.Series(99.0, index=idx), close, 10)
    wide = range_tightness(pd.Series(110.0, index=idx), pd.Series(90.0, index=idx), close, 10)
    assert tight.dropna().iloc[-1] < wide.dropna().iloc[-1]


def test_range_tightness_value() -> None:
    # Rolling range (max high - min low) divided by the current close.
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    out = range_tightness(
        pd.Series(105.0, index=idx),
        pd.Series(95.0, index=idx),
        pd.Series(100.0, index=idx),
        10,
    )
    assert out.dropna().iloc[-1] == pytest.approx(0.1)


def test_volume_ratio_one_for_flat_volume() -> None:
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    vol = pd.Series(1_000_000.0, index=idx)
    out = volume_ratio(vol, 50).dropna()
    assert np.allclose(out, 1.0)


def test_volume_ratio_spikes_above_one() -> None:
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    vol = pd.Series([1_000_000.0] * 59 + [3_000_000.0], index=idx)
    out = volume_ratio(vol, 50)
    assert out.iloc[-1] > 1.5


def test_relative_strength_positive_when_outperforming() -> None:
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    stock = pd.Series(100.0 * (1.02 ** np.arange(60)), index=idx)
    bench = pd.Series(100.0 * (1.01 ** np.arange(60)), index=idx)
    out = relative_strength(stock, bench, 20).dropna()
    assert (out > 0).all()


def test_relative_strength_zero_against_itself(rising_prices: pd.Series) -> None:
    out = relative_strength(rising_prices, rising_prices, 20).dropna()
    assert np.allclose(out, 0.0)
