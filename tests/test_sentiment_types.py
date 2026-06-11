"""Round-trip tests for the sentiment report dataclasses."""

from __future__ import annotations

from tradinglib.sentiment.types import Evidence, SentimentReport, TierReport


def _report() -> SentimentReport:
    return SentimentReport(
        ticker="NVDA",
        as_of="2026-06-11T15:00:00+00:00",
        status="partial",
        tiers=[
            TierReport(
                tier="official",
                label="Official media",
                status="ok",
                score=0.6,
                stance="bullish",
                confidence=0.8,
                summary="Coverage is upbeat.",
                key_themes=["data center demand"],
                evidence=[
                    Evidence(
                        title="Nvidia rallies", source="Reuters", url="https://x", age_days=1.0
                    )
                ],
                metrics={"headline_count": 12},
                source_status={"yfinance_news": "ok", "google_news": "ok"},
                item_count=12,
            ),
            TierReport(tier="forums", label="Serious forums", status="no_data"),
            TierReport(tier="viral", label="Viral / retail", status="degraded", parse_error=True),
        ],
        overall_bias=0.6,
        divergence=None,
    )


def test_roundtrip() -> None:
    report = _report()
    raw = report.to_dict()
    back = SentimentReport.from_dict(raw)
    assert back == report
    assert raw["tiers"][0]["evidence"][0]["source"] == "Reuters"


def test_defaults() -> None:
    tier = TierReport(tier="forums", label="Serious forums", status="no_data")
    assert tier.score is None and tier.evidence == [] and tier.item_count == 0
