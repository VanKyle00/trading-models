"""Earnings-calendar loader (yfinance default provider).

Schema (canonical): a DataFrame with columns
``[ticker, earnings_datetime, session]`` where ``earnings_datetime`` is
UTC-aware and ``session`` is one of ``{"bmo", "amc", "unknown"}``. Cached to
``data/processed/events/earnings/<ticker>/<snapshot>.parquet`` for point-in-time
discipline (the snapshot date is part of the path, never future-leaking into the
data).

yfinance is mocked in tests and never called live (repo convention). The live
method name is version-dependent: ``Ticker.get_earnings_dates(limit=...)`` for
yfinance >= 0.2.x. ``_canonicalize`` assumes ``raw.index`` is a tz-aware
DatetimeIndex (yfinance's 'Earnings Date' index); it falls back to an
'Earnings Date' column if the index is not datetime-like.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "events"
_SUBDIR = "earnings"
_MARKET_OPEN_MIN = 9 * 60 + 30  # 09:30 ET
_MARKET_CLOSE_MIN = 16 * 60  # 16:00 ET


def _session_from_et(ts_utc: pd.Timestamp) -> str:
    """Label a UTC earnings timestamp as before-open / after-close.

    Midnight (yfinance's date-only fallback) is 'unknown'.
    """
    et = ts_utc.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    if minutes == 0:
        return "unknown"
    if minutes <= _MARKET_OPEN_MIN:
        return "bmo"
    if minutes >= _MARKET_CLOSE_MIN:
        return "amc"
    return "unknown"


def _canonicalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance earnings frame into the canonical schema."""
    if pd.api.types.is_datetime64_any_dtype(raw.index):
        source = raw.index
    elif "Earnings Date" in raw.columns:
        source = raw["Earnings Date"]
    else:
        source = raw.index
    dt = pd.to_datetime(source, utc=True)
    df = pd.DataFrame(
        {
            "ticker": ticker,
            "earnings_datetime": pd.DatetimeIndex(dt),
        }
    )
    df["session"] = [_session_from_et(ts) for ts in df["earnings_datetime"]]
    df = df.sort_values("earnings_datetime").reset_index(drop=True)
    return cast(pd.DataFrame, df[["ticker", "earnings_datetime", "session"]])


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "earnings_datetime": pd.Series([], dtype="datetime64[ns, UTC]"),
            "session": pd.Series([], dtype="object"),
        }
    )


def _download_one(ticker: str) -> pd.DataFrame:
    """Fetch and canonicalize the earnings schedule for one ticker."""
    raw = yf.Ticker(ticker).get_earnings_dates(limit=24)
    if raw is None or len(raw) == 0:
        return _empty()
    return _canonicalize(raw, ticker)


def get_earnings_dates(
    tickers: list[str],
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return scheduled earnings for ``tickers`` within ``[start, end]``.

    One parquet cache per ticker keyed by the snapshot (download) date, so a
    cached read never exposes a schedule that postdates its snapshot. Date
    filtering is applied in-memory on the canonical ``earnings_datetime``.
    """
    snapshot = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
        if out.exists() and not refresh:
            df = pd.read_parquet(out)
        else:
            df = _download_one(ticker)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True) if frames else _empty()
    # Defend against pandas downcasting an empty tz-aware column during concat.
    result["earnings_datetime"] = pd.to_datetime(result["earnings_datetime"], utc=True)
    if start is not None:
        result = result[result["earnings_datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        result = result[result["earnings_datetime"] <= pd.Timestamp(end, tz="UTC")]
    return result.reset_index(drop=True)
