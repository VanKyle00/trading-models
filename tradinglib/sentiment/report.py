"""``run_sentiment`` — fetch -> pack -> score -> aggregate for one ticker.

Sources fetch concurrently (thread pool); each failure is recorded per source
and its tier proceeds on whatever remains. Non-empty tiers are scored
concurrently (one provider call each). Aggregation is deterministic: overall
bias is the mean of available tier scores, and a divergence callout fires when
two tiers disagree by >= ``DIVERGENCE_GAP``. The finished report is cached to
``data/processed/sentiment/reports/<ticker>/<date>.json`` — same-day lookups
are served from that JSON (``refresh=True`` bypasses), and the directory
accrues the forward history a future nightly batch job needs.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any

import pandas as pd

from tradinglib.assistant.provider import ClaudeProvider, LLMProvider
from tradinglib.data.paths import processed_dir
from tradinglib.loaders.forums.reddit import get_reddit_posts
from tradinglib.loaders.forums.seeking_alpha import get_seeking_alpha
from tradinglib.loaders.news.google_news import get_google_news
from tradinglib.loaders.news.yfinance import get_news
from tradinglib.loaders.sentiment.google_trends import load_interest
from tradinglib.loaders.social.bluesky import get_bluesky_posts
from tradinglib.loaders.social.stocktwits import get_stocktwits
from tradinglib.sentiment import packs, scoring
from tradinglib.sentiment.types import (
    DIVERGENCE_GAP,
    TIER_FORUMS,
    TIER_LABELS,
    TIER_OFFICIAL,
    TIER_VIRAL,
    Evidence,
    SentimentReport,
    TierReport,
)

SOURCE = "sentiment"
_REPORTS = "reports"
SERIOUS_SUBREDDITS = ("stocks", "investing", "ValueInvesting", "SecurityAnalysis")
VIRAL_SUBREDDITS = ("wallstreetbets",)
_TRENDS_WINDOW_DAYS = 97  # 7-day spike window vs ~90-day baseline

logger = logging.getLogger(__name__)


def _trends_series(ticker: str, *, refresh: bool) -> pd.Series:
    now = pd.Timestamp.now("UTC")
    start = (now - pd.Timedelta(days=_TRENDS_WINDOW_DAYS)).strftime("%Y-%m-%d")
    return load_interest(
        f"{ticker} stock", timeframe=f"{start} {now.strftime('%Y-%m-%d')}", refresh=refresh
    )


def _fetch_sources(ticker: str, *, refresh: bool) -> tuple[dict[str, Any], dict[str, str]]:
    jobs = {
        "yfinance_news": lambda: get_news(ticker, max_items=15, refresh=refresh),
        "google_news": lambda: get_google_news(ticker, max_items=25, refresh=refresh),
        "seeking_alpha": lambda: get_seeking_alpha(ticker, max_items=20, refresh=refresh),
        "reddit": lambda: get_reddit_posts(
            ticker, SERIOUS_SUBREDDITS, limit=10, refresh=refresh
        ).head(20),
        "wsb": lambda: get_reddit_posts(ticker, VIRAL_SUBREDDITS, limit=20, refresh=refresh).head(
            20
        ),
        "bluesky": lambda: get_bluesky_posts(ticker, max_items=25, refresh=refresh),
        "stocktwits": lambda: get_stocktwits(ticker, max_items=30, refresh=refresh),
        "google_trends": lambda: _trends_series(ticker, refresh=refresh),
    }
    data: dict[str, Any] = {}
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, future in futures.items():
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("sentiment source %s failed for %s: %s", name, ticker, exc)
                data[name] = None
                status[name] = f"error: {exc}"
                continue
            data[name] = result
            status[name] = "ok" if len(result) > 0 else "empty"
    return data, status


def _frame(data: dict[str, Any], key: str) -> pd.DataFrame:
    value = data.get(key)
    return value if value is not None else pd.DataFrame()


def _tier_status(item_count: int, score: dict, source_status: dict[str, str]) -> str:
    if item_count == 0:
        return "no_data"
    if score["parse_error"] or score["score"] is None:
        return "degraded"
    # google_trends is a supplementary metric, not pack content — its failures stay visible in source_status but don't degrade the tier (spec: "degrade silently")
    if any(k != "google_trends" and v.startswith("error") for k, v in source_status.items()):
        return "degraded"
    return "ok"


def _evidence(score: dict, kept: list[dict]) -> list[Evidence]:
    out: list[Evidence] = []
    for i in score["evidence_indices"]:
        if i < len(kept):
            item = kept[i]
            url = str(item.get("url", ""))
            if not url.startswith(("http://", "https://")):
                url = ""  # scheme allowlist: open-web feeds could carry javascript: links
            out.append(
                Evidence(
                    title=str(item["title"])[:160],
                    source=str(item["source"]),
                    url=url,
                    age_days=packs.age_days(item.get("published")),
                )
            )
    return out


def _divergence(tier_scores: dict[str, float]) -> dict[str, Any] | None:
    if len(tier_scores) < 2:
        return None
    hi = max(tier_scores, key=lambda k: tier_scores[k])
    lo = min(tier_scores, key=lambda k: tier_scores[k])
    gap = round(tier_scores[hi] - tier_scores[lo], 3)
    if gap < DIVERGENCE_GAP:
        return None
    return {"pair": [hi, lo], "gap": gap}


def run_sentiment(
    ticker: str, *, refresh: bool = False, provider: LLMProvider | None = None
) -> SentimentReport:
    """Three-tier sentiment read for one ticker; cached per (ticker, UTC day)."""
    ticker = ticker.strip().upper()
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _REPORTS / ticker / f"{snapshot}.json"
    if out.exists() and not refresh:
        try:
            return SentimentReport.from_dict(json.loads(out.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("corrupt sentiment report cache %s; refetching", out)

    data, status = _fetch_sources(ticker, refresh=refresh)
    trends = data.get("google_trends")  # Series or None

    tiers_def = [
        (
            TIER_OFFICIAL,
            packs.news_items(_frame(data, "yfinance_news"), _frame(data, "google_news")),
            scoring.official_metrics(_frame(data, "yfinance_news"), _frame(data, "google_news")),
            {k: status[k] for k in ("yfinance_news", "google_news")},
        ),
        (
            TIER_FORUMS,
            packs.forum_items(_frame(data, "seeking_alpha"), _frame(data, "reddit")),
            scoring.forum_metrics(_frame(data, "seeking_alpha"), _frame(data, "reddit")),
            {k: status[k] for k in ("seeking_alpha", "reddit")},
        ),
        (
            TIER_VIRAL,
            packs.viral_items(
                _frame(data, "wsb"), _frame(data, "stocktwits"), _frame(data, "bluesky")
            ),
            scoring.viral_metrics(
                _frame(data, "wsb"),
                _frame(data, "stocktwits"),
                _frame(data, "bluesky"),
                trends,
            ),
            {k: status[k] for k in ("wsb", "stocktwits", "bluesky", "google_trends")},
        ),
    ]

    built: dict[str, dict[str, Any]] = {}
    for tier_id, items, metrics, src_status in tiers_def:
        metric_lines = [f"{k}: {'n/a' if v is None else v}" for k, v in metrics.items()]
        pack, kept = packs.build_pack(ticker, TIER_LABELS[tier_id], items, metric_lines)
        built[tier_id] = {"pack": pack, "kept": kept, "metrics": metrics, "src": src_status}

    to_score = [tier_id for tier_id, b in built.items() if b["kept"]]
    llm_error: str | None = None
    if to_score and provider is None:
        try:
            provider = ClaudeProvider()
        except Exception as exc:
            llm_error = f"LLM scoring unavailable: {exc}"

    score_results: dict[str, dict] = {}
    if to_score and provider is not None:
        with ThreadPoolExecutor(max_workers=len(to_score)) as pool:
            futures = {
                tier_id: pool.submit(scoring.score_tier, provider, tier_id, built[tier_id]["pack"])
                for tier_id in to_score
            }
            score_results = {tier_id: f.result() for tier_id, f in futures.items()}
    for tier_id in to_score:
        if tier_id not in score_results:
            score_results[tier_id] = scoring.unavailable(llm_error or "LLM scoring unavailable")

    tier_reports: list[TierReport] = []
    for tier_id, _items, _metrics, _src in tiers_def:
        b = built[tier_id]
        s = score_results.get(tier_id, scoring.empty_scoring())
        tier_reports.append(
            TierReport(
                tier=tier_id,
                label=TIER_LABELS[tier_id],
                status=_tier_status(len(b["kept"]), s, b["src"]),
                score=s["score"],
                stance=s["stance"],
                confidence=s["confidence"],
                summary=s["summary"],
                key_themes=s["key_themes"],
                evidence=_evidence(s, b["kept"]),
                metrics=b["metrics"],
                source_status=b["src"],
                item_count=len(b["kept"]),
                parse_error=s["parse_error"],
            )
        )

    tier_scores = {t.tier: t.score for t in tier_reports if t.score is not None}
    statuses = {t.status for t in tier_reports}
    report = SentimentReport(
        ticker=ticker,
        as_of=pd.Timestamp.now("UTC").isoformat(),
        status="no_data"
        if statuses == {"no_data"}
        else ("ok" if statuses == {"ok"} else "partial"),
        tiers=tier_reports,
        overall_bias=round(mean(tier_scores.values()), 3) if tier_scores else None,
        divergence=_divergence(tier_scores),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(out)  # atomic; a crash mid-write can't leave a corrupt cache
    return report
