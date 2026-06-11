"""Bounded per-tier text packs from loader rows.

A pack is a numbered list of one-line items -- ``[i] (source, age) text`` --
under a small header carrying the tier's mechanical metrics. The kept-item
list is returned alongside the text so the LLM's ``evidence_indices`` resolve
back to real rows (the model never emits URLs, so links can't be hallucinated).
"""

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any

import pandas as pd

_ITEM_MAX_CHARS = 280
_PACK_MAX_CHARS = 10_000


def _str(value: Any) -> str:
    """Coerce a possibly-NaN/None pandas scalar to str ('' when not a string)."""
    return value if isinstance(value, str) else ""


def age_days(published: Any) -> float | None:
    """Days since ``published`` (UTC), rounded to 0.1; None when unknown."""
    ts = pd.to_datetime(published, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    age = (pd.Timestamp.now("UTC") - ts).total_seconds() / 86_400.0
    return round(max(age, 0.0), 1)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    for item in items:
        key = _norm_title(str(item["title"])) or str(item["text"])[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def news_items(yf_news: pd.DataFrame, google_news: pd.DataFrame) -> list[dict]:
    """Tier-1 items: yfinance headlines (title+summary) + Google News titles."""
    items: list[dict] = []
    for r in yf_news.itertuples():
        summary = _str(r.summary)
        text = f"{r.title} — {summary}" if summary else r.title
        items.append(
            {
                "source": _str(r.publisher) or "Yahoo Finance",
                "title": r.title,
                "text": text,
                "url": "",
                "published": r.published,
            }
        )
    for r in google_news.itertuples():
        items.append(
            {
                "source": _str(r.publisher) or "Google News",
                "title": r.title,
                "text": r.title,
                "url": r.url,
                "published": r.published,
            }
        )
    return _dedupe(items)


def _reddit_item(r: Any) -> dict:
    body = _str(r.text)
    text = f"{r.title} — {body}" if body else r.title
    return {
        "source": f"r/{r.subreddit} (+{int(r.score)}, {int(r.num_comments)}c)",
        "title": r.title,
        "text": text,
        "url": r.url,
        "published": r.created,
    }


def forum_items(seeking_alpha: pd.DataFrame, reddit_posts: pd.DataFrame) -> list[dict]:
    """Tier-2 items: Seeking Alpha titles + serious-subreddit posts."""
    items: list[dict] = [
        {
            "source": "Seeking Alpha",
            "title": r.title,
            "text": r.title,
            "url": r.url,
            "published": r.published,
        }
        for r in seeking_alpha.itertuples()
    ]
    items.extend(_reddit_item(r) for r in reddit_posts.itertuples())
    return _dedupe(items)


def viral_items(
    wsb_posts: pd.DataFrame, stocktwits: pd.DataFrame, bluesky: pd.DataFrame
) -> list[dict]:
    """Tier-3 items: r/wallstreetbets + Stocktwits + Bluesky, round-robin
    interleaved so the bounded pack can't crowd any single source out."""
    wsb = [_reddit_item(r) for r in wsb_posts.itertuples()]
    st: list[dict] = []
    for r in stocktwits.itertuples():
        # NaN-safe: a non-normalized frame can carry float NaN, which is truthy
        tag = f" [user-tagged {_str(r.sentiment)}]" if _str(r.sentiment) else ""
        st.append(
            {
                "source": "Stocktwits",
                "title": str(r.body)[:80],
                "text": f"{r.body}{tag}",
                "url": r.url,
                "published": r.created,
            }
        )
    bsky: list[dict] = []
    for r in bluesky.itertuples():
        text = _str(r.text)
        bsky.append(
            {
                "source": f"Bluesky @{_str(r.handle)} (+{int(r.likes)}, {int(r.reposts)}r)",
                "title": text[:80],
                "text": text,
                "url": r.url,
                "published": r.created,
            }
        )
    # Round-robin so build_pack's char budget trims all sources evenly. Dedupe
    # stays cross-source on purpose: the same message amplified on two networks
    # is one signal, not two (first-seen wins).
    interleaved = [item for trio in zip_longest(wsb, st, bsky) for item in trio if item is not None]
    return _dedupe(interleaved)


def build_pack(
    ticker: str, tier_label: str, items: list[dict], metric_lines: list[str]
) -> tuple[str, list[dict]]:
    """Assemble the bounded numbered pack; returns ``(pack_text, kept_items)``."""
    header = [f"# {ticker} — {tier_label} sentiment pack", *metric_lines, ""]
    lines = list(header)
    kept: list[dict] = []
    used = sum(len(line) + 1 for line in header)
    for item in items:
        text = " ".join(str(item["text"]).split())[:_ITEM_MAX_CHARS]
        age = age_days(item.get("published"))
        age_part = f", {age:g}d" if age is not None else ""
        line = f"[{len(kept)}] ({item['source']}{age_part}) {text}"
        if used + len(line) + 1 > _PACK_MAX_CHARS:
            break
        lines.append(line)
        kept.append(item)
        used += len(line) + 1
    return "\n".join(lines), kept
