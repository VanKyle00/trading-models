"""Fundamentals-snapshot loader (yfinance ``Ticker.info`` provider).

Schema (canonical): one row per ticker with columns
``[ticker, snapshot, market_cap, total_revenue, revenue_growth,
earnings_growth, operating_margin, debt_to_equity, free_cashflow,
forward_pe, trailing_pe, roe, avg_volume]`` — all metrics nullable floats.
Missing ``.info`` keys become NaN and a ticker whose fetch raises yields an
all-NaN row, so one bad name never kills a 500-ticker scan.

Cached to ``data/processed/fundamentals/yfinance/<ticker>/<snapshot>.parquet``
(point-in-time snapshot discipline, same pattern as the earnings loader).
yfinance is mocked in tests and never called live (repo convention).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "fundamentals"
_SUBDIR = "yfinance"

# canonical column -> yfinance .info key
_FIELDS: dict[str, str] = {
    "market_cap": "marketCap",
    "total_revenue": "totalRevenue",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "operating_margin": "operatingMargins",
    "debt_to_equity": "debtToEquity",
    "free_cashflow": "freeCashflow",
    "forward_pe": "forwardPE",
    "trailing_pe": "trailingPE",
    "roe": "returnOnEquity",
    "avg_volume": "averageVolume",
}

logger = logging.getLogger(__name__)


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _download_one(ticker: str, snapshot: str) -> pd.DataFrame:
    """Fetch one ticker's ``.info`` and map it to a one-row canonical frame."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        logger.warning("fundamentals fetch failed for %s; emitting NaN row", ticker, exc_info=True)
        info = {}
    row: dict[str, object] = {"ticker": ticker, "snapshot": snapshot}
    for column, key in _FIELDS.items():
        row[column] = _to_float(info.get(key, np.nan))
    return pd.DataFrame([row])


def get_fundamental_snapshot(
    tickers: list[str],
    *,
    refresh: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Return a one-row-per-ticker fundamentals snapshot for ``tickers``.

    Cached reads are free for the rest of the day; uncached tickers are
    fetched on a small thread pool (`.info` is one HTTP round-trip each).
    """
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    cached: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
        if out.exists() and not refresh:
            cached[ticker] = pd.read_parquet(out)
        else:
            missing.append(ticker)

    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fetched = list(pool.map(lambda t: _download_one(t, snapshot), missing))
        for ticker, df in zip(missing, fetched, strict=True):
            out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            cached[ticker] = df

    frames = [cached[ticker] for ticker in tickers]
    return pd.concat(frames, ignore_index=True)
