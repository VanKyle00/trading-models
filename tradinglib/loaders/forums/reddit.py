"""Reddit search loader — posts mentioning a ticker in given subreddits.

Serves both sentiment tiers: the engine passes serious-investing subreddits
for Tier 2 and r/wallstreetbets for Tier 3 (the tier->subreddit mapping lives
in ``tradinglib/sentiment/report.py``, not here).

Schema (canonical): ``[ticker, subreddit, created, title, text, score,
num_comments, url]``, UTC-aware, newest first. ``text`` is the selftext capped
at 500 chars. Cached per subreddit to
``data/processed/forums/reddit/<subreddit>/<ticker>/<snapshot>.parquet`` —
cache hits need no credentials.

Credentials: free OAuth app (script type) from reddit.com/prefs/apps via
``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` (+ optional
``REDDIT_USER_AGENT``). Missing credentials raise
``MissingRedditCredentials`` on uncached fetches; the sentiment engine
catches it and degrades the tier rather than crashing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "forums"
_SUBDIR = "reddit"
_TEXT_MAX_CHARS = 500

logger = logging.getLogger(__name__)


class MissingRedditCredentials(RuntimeError):  # noqa: N818
    """REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not configured."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "subreddit": pd.Series([], dtype="object"),
            "created": pd.Series([], dtype="datetime64[ms, UTC]"),
            "title": pd.Series([], dtype="object"),
            "text": pd.Series([], dtype="object"),
            "score": pd.Series([], dtype="int64"),
            "num_comments": pd.Series([], dtype="int64"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _make_reddit() -> Any:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise MissingRedditCredentials(
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set "
            "(create a script app at reddit.com/prefs/apps)"
        )
    import praw  # lazy: only needed for uncached fetches with credentials

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=os.environ.get("REDDIT_USER_AGENT", "trading-models-sentiment/0.1"),
    )


def _download_sub(reddit: Any, ticker: str, sub: str, limit: int) -> pd.DataFrame:
    try:
        submissions = list(
            reddit.subreddit(sub).search(
                f"{ticker} OR ${ticker}", sort="relevance", time_filter="week", limit=limit
            )
        )
    except Exception:
        logger.warning("reddit search failed for %s in r/%s; empty", ticker, sub, exc_info=True)
        submissions = []
    rows = [
        {
            "subreddit": sub,
            "created": pd.Timestamp(s.created_utc, unit="s", tz="UTC"),
            "title": s.title,
            "text": (s.selftext or "")[:_TEXT_MAX_CHARS],
            "score": int(s.score),
            "num_comments": int(s.num_comments),
            "url": f"https://www.reddit.com{s.permalink}",
        }
        for s in submissions
    ]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    # ms (not ns) so the dtype survives the parquet round-trip and cached == fresh
    df["created"] = pd.to_datetime(df["created"], utc=True).astype("datetime64[ms, UTC]")
    return df


def get_reddit_posts(
    ticker: str, subreddits: tuple[str, ...], *, limit: int = 20, refresh: bool = False
) -> pd.DataFrame:
    """Posts mentioning ``ticker`` across ``subreddits`` (last week), newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    reddit: Any = None
    for sub in subreddits:
        out = processed_dir(SOURCE) / _SUBDIR / sub / ticker / f"{snapshot}.parquet"
        if out.exists() and not refresh:
            frames.append(pd.read_parquet(out))
            continue
        if reddit is None:
            reddit = _make_reddit()  # raises MissingRedditCredentials when unconfigured
        df = _download_sub(reddit, ticker, sub, limit)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        frames.append(df)
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return _empty()
    combined = pd.concat(non_empty, ignore_index=True)
    return combined.sort_values("created", ascending=False).reset_index(drop=True)
