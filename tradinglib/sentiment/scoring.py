"""Per-tier LLM scoring + mechanical (no-LLM) metrics.

``score_tier`` mirrors ``scanner/briefs.py``: ONE provider call demanding
strict JSON, hard sanitization, graceful degradation (malformed JSON keeps the
raw text as the summary with ``score=None``) — it never raises. Mechanical
metrics are plain-code facts computed from loader rows, independent of the LLM.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from tradinglib.assistant.provider import LLMProvider
from tradinglib.assistant.types import UserMsg
from tradinglib.sentiment.types import TIER_FORUMS, TIER_OFFICIAL, TIER_VIRAL

_STANCES = ("bearish", "neutral", "bullish", "mixed")
_MAX_THEMES = 5
_MAX_EVIDENCE = 5
_RAW_SUMMARY_MAX_CHARS = 600

_TIER_DESC = {
    TIER_OFFICIAL: "Tier-1 financial news headlines",
    TIER_FORUMS: "serious investor forums (Seeking Alpha article titles, finance subreddits)",
    TIER_VIRAL: "viral retail chatter (r/wallstreetbets posts, Stocktwits messages)",
}

SCORE_SYSTEM_PROMPT = (
    "You are a market-sentiment analyst. You receive a pack of recent text items "
    "about one stock, all from a single source tier — {tier_desc}. Items are "
    "numbered like [3]; the header lists mechanical metrics for context. Judge "
    "the aggregate sentiment of the pack toward the stock over the coming weeks. "
    "Respond with ONLY a JSON object, no prose before or after, with exactly "
    "these keys: score (number, -1 strongly bearish to 1 strongly bullish), "
    "stance (one of: bearish, neutral, bullish, mixed — mixed means genuinely "
    "conflicting reads, not merely thin data), confidence (number 0-1: how "
    "substantive and consistent the pack is), summary (string, 2-3 sentences), "
    "key_themes (array of at most 5 short strings), evidence_indices (array of "
    "at most 5 integers — the item numbers that most drove your judgement). "
    "Ground every claim in the pack; do not invent facts."
)


def _clamp(value: Any, lo: float, hi: float) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return min(max(f, lo), hi)


def _sanitize(raw: dict, *, parse_error: bool = False) -> dict:
    stance = raw.get("stance")
    if stance not in _STANCES:
        stance = None
    indices: list[int] = []
    for x in raw.get("evidence_indices") or []:
        if isinstance(x, int) and not isinstance(x, bool) and x >= 0 and x not in indices:
            indices.append(x)
    return {
        "score": _clamp(raw.get("score"), -1.0, 1.0),
        "stance": stance,
        "confidence": _clamp(raw.get("confidence"), 0.0, 1.0),
        "summary": str(raw.get("summary", "")),
        "key_themes": [str(x) for x in (raw.get("key_themes") or [])][:_MAX_THEMES],
        "evidence_indices": indices[:_MAX_EVIDENCE],
        "parse_error": parse_error,
    }


def empty_scoring() -> dict:
    """Neutral scoring shell for tiers with no content (no LLM call made)."""
    return _sanitize({})


def unavailable(reason: str) -> dict:
    """Scoring stand-in when the LLM cannot run (no key, provider down)."""
    return _sanitize({"summary": reason}, parse_error=True)


def score_tier(provider: LLMProvider, tier: str, pack: str) -> dict:
    """One provider call -> sanitized tier-scoring dict; never raises."""
    system = SCORE_SYSTEM_PROMPT.format(tier_desc=_TIER_DESC[tier])
    try:
        turn = provider.complete(system, [UserMsg(pack)], tools=[])
    except Exception as exc:
        return unavailable(f"LLM scoring unavailable: {exc}")
    text = turn.text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return _sanitize(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    return _sanitize({"summary": text[:_RAW_SUMMARY_MAX_CHARS]}, parse_error=True)


def official_metrics(yf_news: pd.DataFrame, google_news: pd.DataFrame) -> dict:
    return {"headline_count": int(len(yf_news) + len(google_news))}


def forum_metrics(seeking_alpha: pd.DataFrame, reddit_posts: pd.DataFrame) -> dict:
    metrics: dict[str, Any] = {"post_count": int(len(seeking_alpha) + len(reddit_posts))}
    if len(reddit_posts) > 0:
        metrics["mean_upvotes"] = round(float(reddit_posts["score"].mean()), 1)
        metrics["mean_comments"] = round(float(reddit_posts["num_comments"].mean()), 1)
    else:
        metrics["mean_upvotes"] = None
        metrics["mean_comments"] = None
    return metrics


def trends_spike(interest: pd.Series | None) -> float | None:
    """mean(last 7 days) / mean(prior window); None when either side is missing."""
    if interest is None or len(interest) == 0:
        return None
    cutoff = interest.index.max() - pd.Timedelta(days=7)
    recent = interest[interest.index > cutoff]
    base = interest[interest.index <= cutoff]
    if len(recent) == 0 or len(base) == 0 or float(base.mean()) <= 0.0:
        return None
    return round(float(recent.mean()) / float(base.mean()), 2)


def viral_metrics(
    wsb_posts: pd.DataFrame, stocktwits: pd.DataFrame, interest: pd.Series | None
) -> dict:
    bulls = int((stocktwits["sentiment"] == "Bullish").sum()) if len(stocktwits) else 0
    bears = int((stocktwits["sentiment"] == "Bearish").sum()) if len(stocktwits) else 0
    return {
        "wsb_mentions": len(wsb_posts),
        "st_messages": len(stocktwits),
        "st_bullish": bulls,
        "st_bearish": bears,
        "st_bull_bear_ratio": round(bulls / bears, 2) if bears > 0 else None,
        "trends_spike": trends_spike(interest),
    }
