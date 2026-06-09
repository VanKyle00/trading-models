"""News-headline loader (yfinance ``Ticker.news`` provider).

Schema (canonical): ``[ticker, published, title, summary, publisher]`` with
``published`` UTC-aware, newest first, capped at ``max_items``. Headlines and
summaries only — no article bodies — so the LLM doc pack stays small.
Snapshot-cached to ``data/processed/news/yfinance/<ticker>/<snapshot>.parquet``.

yfinance has shipped two news shapes: a flat dict (``title``, ``publisher``,
``providerPublishTime``) and a nested one (``content`` -> ``title``,
``summary``, ``pubDate``, ``provider.displayName``); both are handled.
yfinance is mocked in tests and never called live (repo convention).
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "news"
_SUBDIR = "yfinance"

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "published": pd.Series([], dtype="datetime64[ns, UTC]"),
            "title": pd.Series([], dtype="object"),
            "summary": pd.Series([], dtype="object"),
            "publisher": pd.Series([], dtype="object"),
        }
    )


def _canonicalize_item(item: dict) -> dict | None:
    if "content" in item:  # nested (newer yfinance)
        content = item.get("content") or {}
        published = pd.to_datetime(content.get("pubDate"), utc=True, errors="coerce")
        return {
            "published": published,
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "publisher": (content.get("provider") or {}).get("displayName", ""),
        }
    if "title" in item:  # flat (older yfinance)
        epoch = item.get("providerPublishTime")
        published = pd.Timestamp(epoch, unit="s", tz="UTC") if epoch else pd.NaT
        return {
            "published": published,
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "publisher": item.get("publisher", ""),
        }
    return None


def _download_one(ticker: str) -> pd.DataFrame:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        logger.warning("news fetch failed for %s; returning empty", ticker, exc_info=True)
        items = []
    rows = [r for r in (_canonicalize_item(i) for i in items) if r is not None]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df.sort_values("published", ascending=False).reset_index(drop=True)


def get_news(ticker: str, *, max_items: int = 15, refresh: bool = False) -> pd.DataFrame:
    """Recent headlines for one ticker, newest first, capped at ``max_items``."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download_one(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
