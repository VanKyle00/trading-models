"""Technical features computed from a price series.

All functions take a ``pd.Series`` of prices and return a ``pd.Series`` of
the same length, with leading values that lack enough history set to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over ``periods`` bars: ln(p_t / p_{t-periods})."""
    return np.log(prices / prices.shift(periods))


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling sample standard deviation of returns."""
    return returns.rolling(window).std()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using simple moving averages of gains/losses.

    The classic Wilder formulation uses an exponential smoothing; the SMA
    version is close enough for feature-engineering purposes and matches
    most charting platforms' default.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    # Add a tiny epsilon so a pure-uptrend window (avg_loss == 0) saturates
    # the index near 100 rather than producing NaN.
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def price_to_sma_ratio(prices: pd.Series, window: int) -> pd.Series:
    """Deviation of price from its ``window``-bar SMA, expressed as a ratio.

    Values > 0 mean price is above the SMA; values < 0 mean below.
    """
    return prices / prices.rolling(window).mean() - 1.0


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` bars."""
    return prices.rolling(window).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range: rolling mean of the true range.

    True range is the largest of (high - low), |high - prev close| and
    |low - prev close|, so overnight gaps count toward the range. The first
    bar has no previous close and falls back to (high - low).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def ma_slope(prices: pd.Series, window: int, lookback: int) -> pd.Series:
    """Fractional change of the ``window``-bar SMA over ``lookback`` bars.

    Positive values mean the moving average is rising.
    """
    avg = prices.rolling(window).mean()
    return avg / avg.shift(lookback) - 1.0


def pct_off_52w_high(prices: pd.Series, window: int = 252) -> pd.Series:
    """Distance from the rolling ``window``-bar high, as a ratio <= 0.

    0 means the price is at a new rolling high; -0.2 means 20% below it.
    """
    return prices / prices.rolling(window).max() - 1.0


def range_tightness(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Rolling price range (max high - min low) relative to the current close.

    Small values indicate a tight consolidation.
    """
    rng = high.rolling(window).max() - low.rolling(window).min()
    return rng / close


def volume_ratio(volume: pd.Series, window: int = 50) -> pd.Series:
    """Volume relative to its trailing ``window``-bar average."""
    return volume / volume.rolling(window).mean()


def relative_strength(
    prices: pd.Series, benchmark_prices: pd.Series, window: int = 126
) -> pd.Series:
    """Trailing ``window``-bar return of ``prices`` minus the benchmark's.

    Positive values mean the asset has outperformed the benchmark over the
    window. The benchmark is aligned on the price index first.
    """
    bench = benchmark_prices.reindex(prices.index)
    return prices.pct_change(window) - bench.pct_change(window)
