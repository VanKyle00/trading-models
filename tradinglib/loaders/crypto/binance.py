"""Binance public klines loader — daily OHLCV for any spot pair.

Uses ``data-api.binance.vision/api/v3/klines``, Binance's public data
mirror. Same response shape as ``api.binance.com`` but globally
accessible (the main API blocks several regions, including the US, with
HTTP 451). No API key required for market data.

Canonical schema (written to parquet):
    timestamp: datetime64[ns, UTC]    (index, open-time of the bar)
    open, high, low, close: float64
    volume: float64                   (base-asset volume)
    symbol: str
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "binance"
BASE_URL = "https://data-api.binance.vision/api/v3/klines"
MAX_LIMIT = 1000  # Binance caps candles per request at 1000

_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def load_daily(
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load daily klines for a Binance spot pair (e.g. ``"BTCUSDT"``).

    The first call paginates through Binance's klines endpoint and caches
    the full result to ``data/processed/binance/<symbol>/daily.parquet``.
    Subsequent calls read the parquet and apply the date filter in-memory.
    """
    out = processed_dir(SOURCE) / symbol / "daily.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download_daily(symbol, start, end)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def _download_daily(
    symbol: str,
    start: str | datetime | None,
    end: str | datetime | None,
) -> pd.DataFrame:
    """Paginate through klines and stitch them into a single DataFrame."""
    start_ms = _to_ms(start) if start is not None else 0
    end_ms = _to_ms(end) if end is not None else None

    rows: list[list] = []
    cursor = start_ms
    with httpx.Client(timeout=30.0) as client:
        while True:
            params: dict[str, int | str] = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "limit": MAX_LIMIT,
            }
            if end_ms is not None:
                params["endTime"] = end_ms
            resp = client.get(BASE_URL, params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < MAX_LIMIT:
                break
            # Advance past the last close_time to avoid re-fetching the same bar
            cursor = int(batch[-1][6]) + 1
            if end_ms is not None and cursor > end_ms:
                break

    if not rows:
        raise ValueError(f"Binance returned no data for {symbol!r}")
    return _canonicalize(rows, symbol)


def _canonicalize(rows: list[list], symbol: str) -> pd.DataFrame:
    """Convert the raw klines array to the canonical DataFrame shape."""
    df = pd.DataFrame(rows, columns=_KLINE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float64")
    df["symbol"] = symbol
    return cast(pd.DataFrame, df[["open", "high", "low", "close", "volume", "symbol"]])


def _to_ms(t: str | datetime) -> int:
    """Convert a date string or datetime to UTC milliseconds since epoch."""
    if isinstance(t, str):
        ts = pd.Timestamp(t, tz="UTC")
    else:
        ts = pd.Timestamp(t)
        if ts.tz is None:
            ts = ts.tz_localize(UTC)
    return int(ts.timestamp() * 1000)
