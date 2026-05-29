"""Binance historical aggTrades loader — tick-level data from the public CDN.

Pulls one daily ZIP file at a time from
``data.binance.vision/data/spot/daily/aggTrades/<symbol>/``. Each ZIP
contains one CSV with all aggregated trades for one symbol-day, no
header row. Files are large (~50-100 MB per BTCUSDT-day in 2024), so
caching is essential. The canonicalized parquet is much smaller because
we drop the per-trade IDs that consumers don't need.

Canonical schema (per parquet, one file per symbol-day):
    timestamp: datetime64[ns, UTC]    (index)
    price: float64
    qty: float64                       (base-asset volume of the aggTrade)
    is_buyer_maker: bool
        ``True``  → trade was a passive bid lift (i.e. a taker SELL)
        ``False`` → trade was an aggressive lift of the ask (i.e. a taker BUY)
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import cast

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "binance"
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"

_RAW_COLUMNS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "timestamp_ms",
    "is_buyer_maker",
    "is_best_match",
]


def load_trades(
    symbol: str,
    start: str | date,
    end: str | date,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load aggregated trades for ``symbol`` between ``start`` and ``end`` (inclusive).

    Each day's data is downloaded, canonicalized, and cached as a separate
    parquet at ``data/processed/binance/<symbol>/aggTrades/YYYY-MM-DD.parquet``.
    Multi-day requests stitch the per-day parquets together in-memory.
    """
    return pd.concat(iter_daily_trades(symbol, start, end, refresh=refresh)).sort_index()


def iter_daily_trades(
    symbol: str,
    start: str | date,
    end: str | date,
    *,
    refresh: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield one day's canonical trades at a time over ``[start, end]`` inclusive.

    The lazy, memory-bounded counterpart to :func:`load_trades`: callers that
    aggregate incrementally (see
    :func:`tradinglib.features.microstructure.aggregate_daily_chunks`) can
    process a multi-day range without ever holding more than a single day's
    millions of ticks in memory. ``load_trades`` is just
    ``pd.concat(iter_daily_trades(...))`` for callers that do want the whole
    range at once.
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    if start_d > end_d:
        raise ValueError(f"start ({start_d}) is after end ({end_d})")

    cur = start_d
    while cur <= end_d:
        yield _load_one_day(symbol, cur, refresh=refresh)
        cur += timedelta(days=1)


def _load_one_day(symbol: str, day: date, *, refresh: bool) -> pd.DataFrame:
    cache = processed_dir(SOURCE) / symbol / "aggTrades" / f"{day.isoformat()}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    df = _download_one_day(symbol, day)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _download_one_day(symbol: str, day: date) -> pd.DataFrame:
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    with httpx.Client(timeout=300.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return _parse_zip(resp.content)


def _parse_zip(content: bytes) -> pd.DataFrame:
    """Unpack a Binance aggTrades ZIP and return the canonical DataFrame."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as f:
            # Only read the four columns we keep — skipping the three int64
            # trade-id columns (and is_best_match) roughly halves peak parse
            # memory on a ~9M-row BTCUSDT-day, which matters on memory-capped
            # hosts. The canonical parquet schema is unchanged.
            df = pd.read_csv(
                f,
                names=_RAW_COLUMNS,
                header=None,
                usecols=["price", "qty", "timestamp_ms", "is_buyer_maker"],
                dtype={
                    "price": "float64",
                    "qty": "float64",
                    "timestamp_ms": "int64",
                    "is_buyer_maker": "str",
                },
            )
    df["is_buyer_maker"] = df["is_buyer_maker"] == "True"
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return cast(pd.DataFrame, df[["price", "qty", "is_buyer_maker"]])


def _to_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()
