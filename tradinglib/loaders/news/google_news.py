"""Google News RSS loader — Tier-1 headline search for one ticker.

Schema (canonical): ``[ticker, published, title, publisher, url]`` with
``published`` UTC-aware, newest first, capped at ``max_items``. Query is
``<TICKER> stock when:14d`` -- the "stock" suffix disambiguates single-letter
tickers (people search "F stock" too) and ``when:`` bounds the window.
Keyless public feed. Snapshot-cached to
``data/processed/news/google_news/<ticker>/<snapshot>.parquet``.
httpx is stubbed in tests (repo convention: no live network).
"""

from __future__ import annotations

import calendar
import logging
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "news"
_SUBDIR = "google_news"
_TIMEOUT_S = 8.0
_WINDOW_DAYS = 14

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "published": pd.Series([], dtype="datetime64[ms, UTC]"),
            "title": pd.Series([], dtype="object"),
            "publisher": pd.Series([], dtype="object"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _entry_published(entry: Any) -> Any:
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        return pd.NaT
    return pd.Timestamp(calendar.timegm(parsed), unit="s", tz="UTC")


def _download(ticker: str) -> pd.DataFrame:
    query = quote_plus(f"{ticker} stock when:{_WINDOW_DAYS}d")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT_S, follow_redirects=True)
        resp.raise_for_status()
        entries = feedparser.parse(resp.text).entries or []
    except Exception:
        logger.warning("google news fetch failed for %s; returning empty", ticker, exc_info=True)
        entries = []
    rows = [
        {
            "published": _entry_published(e),
            "title": getattr(e, "title", ""),
            "publisher": getattr(getattr(e, "source", None), "title", "") or "Google News",
            "url": getattr(e, "link", ""),
        }
        for e in entries
        if getattr(e, "title", "")
    ]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    # ms (not ns) so the dtype survives the parquet round-trip and cached == fresh
    df["published"] = pd.to_datetime(df["published"], utc=True).astype("datetime64[ms, UTC]")
    return df.sort_values("published", ascending=False).reset_index(drop=True)


def get_google_news(ticker: str, *, max_items: int = 25, refresh: bool = False) -> pd.DataFrame:
    """Recent Google News headlines for one ticker, newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
