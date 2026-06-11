"""Typed report objects for the three-tier sentiment read.

``SentimentReport.to_dict()`` must stay JSON-serializable — it is written to
``data/processed/sentiment/reports/<ticker>/<date>.json`` and served verbatim
by the webapp API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TIER_OFFICIAL = "official"
TIER_FORUMS = "forums"
TIER_VIRAL = "viral"
TIER_LABELS = {
    TIER_OFFICIAL: "Official media",
    TIER_FORUMS: "Serious forums",
    TIER_VIRAL: "Viral / retail",
}

# Two tiers whose scores differ by at least this much trigger the divergence callout.
DIVERGENCE_GAP = 0.6


@dataclass
class Evidence:
    title: str
    source: str
    url: str
    age_days: float | None


@dataclass
class TierReport:
    tier: str  # TIER_OFFICIAL | TIER_FORUMS | TIER_VIRAL
    label: str
    status: str  # "ok" | "degraded" | "no_data"
    score: float | None = None  # -1..1
    stance: str | None = None  # "bearish" | "neutral" | "bullish" | "mixed"
    confidence: float | None = None  # 0..1
    summary: str = ""
    key_themes: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    source_status: dict[str, str] = field(default_factory=dict)  # source -> ok|empty|error: …
    item_count: int = 0
    parse_error: bool = False


@dataclass
class SentimentReport:
    ticker: str
    as_of: str  # ISO-8601 UTC
    status: str  # "ok" | "partial" | "no_data"
    tiers: list[TierReport]
    overall_bias: float | None = None  # mean of available tier scores
    divergence: dict[str, Any] | None = None  # {"pair": [hi_tier, lo_tier], "gap": float}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SentimentReport:
        tiers = [
            TierReport(**{**t, "evidence": [Evidence(**e) for e in t.get("evidence") or []]})
            for t in raw.get("tiers") or []
        ]
        return cls(
            ticker=raw["ticker"],
            as_of=raw["as_of"],
            status=raw["status"],
            tiers=tiers,
            overall_bias=raw.get("overall_bias"),
            divergence=raw.get("divergence"),
        )
