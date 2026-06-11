"""yfinance loader — daily OHLCV bars for US equities, ETFs, indices.

yfinance is unofficial and rate-limited but free and key-less, which makes
it the right choice for portfolio prototyping. For higher-quality or
intraday-level data, prefer Polygon or Alpaca (loaders to come).

Canonical schema written to parquet:
    timestamp: datetime64[ns, UTC]   (index)
    open, high, low, close: float64
    volume: int64
    symbol: str
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "yfinance"
_CACHE_FILENAME = "daily.parquet"


def load_daily(
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV bars for one symbol, caching to parquet on first run.

    Parameters
    ----------
    symbol:
        Ticker as understood by yfinance (e.g. ``"SPY"``, ``"AAPL"``, ``"BTC-USD"``).
    start, end:
        Optional date filters applied after caching. ``None`` means
        unbounded — yfinance returns the full available history on a fresh
        download.
    refresh:
        If ``True``, redownload and overwrite the cached parquet.
    """
    out = processed_dir(SOURCE) / symbol / _CACHE_FILENAME
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download_daily(symbol)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

    # yfinance occasionally emits a partial trailing row (volume populated,
    # prices NaN); a bar without prices is unusable downstream, and filtering
    # on the read path also heals caches written before this guard existed.
    df = df.dropna(subset=["open", "high", "low", "close"])

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def _download_daily(symbol: str) -> pd.DataFrame:
    """Hit yfinance and return a canonicalized DataFrame."""
    raw = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r}")
    return _canonicalize(raw, symbol)


def _canonicalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance's output into the canonical schema.

    yfinance returns MultiIndex columns (outer = OHLCV name, inner = ticker)
    even for a single symbol — flatten that. The index is a DatetimeIndex
    with no name in newer yfinance versions; force it to ``timestamp`` and
    make it timezone-aware UTC.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="timestamp")
    df["symbol"] = symbol
    df["volume"] = df["volume"].astype("int64")
    return cast(pd.DataFrame, df[["open", "high", "low", "close", "volume", "symbol"]])
