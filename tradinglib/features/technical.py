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
