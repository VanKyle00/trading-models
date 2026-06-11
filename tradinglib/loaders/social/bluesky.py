"""Bluesky cashtag-search loader (Tier 3 viral source).

Schema (canonical): ``[ticker, created, text, handle, likes, reposts, url]``,
UTC-aware, kept in the API's engagement ("top") order — deliberately NOT
re-sorted by time, the ranking is the virality signal — capped at
``max_items``.

Keyless: ``api.bsky.app``'s searchPosts works unauthenticated (verified
2026-06-11; the deliberately-public host gates search, so this may close
someday). Any fetch failure logs and returns empty — tier 3 proceeds on
Stocktwits + Trends; an authenticated app-password session is the documented
upgrade path (see the design spec). Snapshot-cached to
``data/processed/social/bluesky/<ticker>/<snapshot>.parquet``.
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "social"
_SUBDIR = "bluesky"
_TIMEOUT_S = 8.0
_WINDOW_DAYS = 7
_UA = "Mozilla/5.0 (compatible; trading-models-sentiment/0.1)"

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "created": pd.Series([], dtype="datetime64[ms, UTC]"),
            "text": pd.Series([], dtype="object"),
            "handle": pd.Series([], dtype="object"),
            "likes": pd.Series([], dtype="int64"),
            "reposts": pd.Series([], dtype="int64"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _row(post: dict) -> dict:
    record = post.get("record") or {}
    handle = (post.get("author") or {}).get("handle", "")
    uri = post.get("uri", "")
    rkey = uri.rsplit("/", 1)[-1] if uri else ""
    return {
        "created": pd.to_datetime(record.get("createdAt"), utc=True, errors="coerce"),
        "text": record.get("text", ""),
        "handle": handle,
        "likes": int(post.get("likeCount") or 0),
        "reposts": int(post.get("repostCount") or 0),
        "url": f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else "",
    }


def _download(ticker: str, max_items: int) -> pd.DataFrame:
    since = (pd.Timestamp.now("UTC") - pd.Timedelta(days=_WINDOW_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    try:
        resp = httpx.get(
            "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={
                "q": f"${ticker}",
                "sort": "top",
                "since": since,
                "lang": "en",
                "limit": max_items,
            },
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        resp.raise_for_status()
        posts = resp.json().get("posts") or []
    except Exception:
        logger.warning("bluesky fetch failed for %s; returning empty", ticker, exc_info=True)
        posts = []
    rows = [_row(p) for p in posts if (p.get("record") or {}).get("text")]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    # ms (not ns) so the dtype survives the parquet round-trip and cached == fresh
    df["created"] = pd.to_datetime(df["created"], utc=True).astype("datetime64[ms, UTC]")
    return df  # keep the API's engagement order — the ranking IS the signal


def get_bluesky_posts(ticker: str, *, max_items: int = 25, refresh: bool = False) -> pd.DataFrame:
    """Top cashtag posts for one ticker over the last week, engagement-ranked."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker, max_items)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
