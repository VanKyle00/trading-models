"""Seeking Alpha per-ticker RSS loader — article titles only (Tier 2).

Schema (canonical): ``[ticker, published, title, url]``, UTC-aware, newest
first, capped at ``max_items``. Seeking Alpha has no public API; this is its
public per-ticker RSS feed (titles + links, no bodies). It is the most fragile
source in the sentiment tool — Cloudflare moods — so any failure logs and
returns empty (tier 2 then runs on Reddit alone). Snapshot-cached to
``data/processed/forums/seeking_alpha/<ticker>/<snapshot>.parquet``.
"""

from __future__ import annotations

import calendar
import logging
from typing import Any

import feedparser
import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "forums"
_SUBDIR = "seeking_alpha"
_TIMEOUT_S = 8.0
_UA = "Mozilla/5.0 (compatible; trading-models-sentiment/0.1)"

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "published": pd.Series([], dtype="datetime64[ms, UTC]"),
            "title": pd.Series([], dtype="object"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _entry_published(entry: Any) -> Any:
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        return pd.NaT
    return pd.Timestamp(calendar.timegm(parsed), unit="s", tz="UTC")


def _download(ticker: str) -> pd.DataFrame:
    url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
    try:
        resp = httpx.get(
            url, timeout=_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": _UA}
        )
        resp.raise_for_status()
        entries = feedparser.parse(resp.text).entries or []
    except Exception:
        logger.warning("seeking alpha fetch failed for %s; returning empty", ticker, exc_info=True)
        entries = []
    rows = [
        {
            "published": _entry_published(e),
            "title": getattr(e, "title", ""),
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


def get_seeking_alpha(ticker: str, *, max_items: int = 20, refresh: bool = False) -> pd.DataFrame:
    """Recent Seeking Alpha article titles for one ticker, newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
