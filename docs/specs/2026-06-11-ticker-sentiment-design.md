# Ticker Sentiment — Three-Tier Read — Design Spec

**Date:** 2026-06-11
**Status:** Approved (pending written-spec review)
**Scope:** An on-demand sentiment tool: type a ticker on a new `/sentiment` webapp page
and get a three-tier read — official media, serious forums, viral/retail — each tier
scored by one bounded LLM call plus mechanical metrics, with a deterministic aggregate
and a tier-divergence callout. Free data sources only. The engine is pure
`fetch → score` so a future nightly cron can loop it over scanner candidates unchanged.

## Motivation

The scanner's FA gate covers fundamentals; nothing covers *narrative*. The same ticker
can read very differently in Tier-1 news, on serious investor forums, and in viral
retail chatter — and the divergence between those reads (e.g., viral hot while official
is cold) is itself information when sizing or timing a swing/options ticket. This tool
makes that three-way read a single lookup.

## Decisions (locked during brainstorming, 2026-06-11)

- **On-demand now, batch later.** v1 is a single-ticker research view. The engine is
  designed so the nightly Modal cron can later call the same `run_sentiment()` over
  scanner candidates; no engine changes anticipated for batch mode.
- **Free sources only.** No X (~$200/mo) and no TikTok (no viable content API at any
  price). The viral tier is proxied by r/wallstreetbets + Stocktwits + Google Trends
  spikes. Matches the repo's no-paid-data precedent (options frictions SP2).
- **Surface: `/sentiment` webapp page only.** Assistant chat tool, CLI script, and
  planner integration are explicitly deferred.
- **Architecture: per-tier adapter pipeline.** Source loaders → bounded per-tier text
  pack → one LLM call per tier (the `scanner/briefs.py` strict-JSON pattern) →
  deterministic aggregation + divergence flag. Tiers degrade independently. The
  single-combined-LLM-call and mechanical-only alternatives were rejected (context
  crowding / no directional read on text, respectively).

## Tier composition

| Tier | Label | Sources | Mechanical metrics (no LLM) |
|------|-------|---------|------------------------------|
| 1 | Official media | yfinance news (existing loader) + Google News RSS | headline count (fetched window) |
| 2 | Serious forums | Seeking Alpha per-ticker RSS (titles) + r/stocks, r/investing, r/ValueInvesting, r/SecurityAnalysis | post count (SA+Reddit), mean upvotes/comments (Reddit) |
| 3 | Viral / retail | r/wallstreetbets + Stocktwits symbol stream + Google Trends (existing loader) | Stocktwits bull/bear ratio (user-tagged), WSB mention count, Trends spike ratio |

One Reddit loader serves tiers 2 and 3, parameterized by subreddit list; the engine
owns the tier→subreddit mapping.

## Components

```
tradinglib/loaders/news/google_news.py      NEW  Google News RSS search (keyless)
tradinglib/loaders/forums/__init__.py       NEW
tradinglib/loaders/forums/seeking_alpha.py  NEW  SA per-ticker RSS, titles only
tradinglib/loaders/forums/reddit.py         NEW  Reddit OAuth search (praw), subreddit list param
tradinglib/loaders/social/__init__.py       NEW
tradinglib/loaders/social/stocktwits.py     NEW  ST symbol stream w/ Bullish/Bearish tags (keyless)

tradinglib/sentiment/__init__.py            NEW
tradinglib/sentiment/types.py               NEW  TierReport / SentimentReport dataclasses
tradinglib/sentiment/packs.py               NEW  bounded per-tier text packs from loader rows
tradinglib/sentiment/scoring.py             NEW  per-tier LLM call + sanitization + mechanical metrics
tradinglib/sentiment/report.py              NEW  run_sentiment(ticker, *, refresh, provider) orchestrator

webapp/sentiment.py                         NEW  view helper (engine call + report shaping)
webapp/templates/sentiment.html             NEW  page template
webapp/main.py                              MOD  GET /sentiment + GET /api/v1/sentiment/{ticker}
```

Existing reused: `loaders/news/yfinance.py`, `loaders/sentiment/google_trends.py`,
`loaders/equities/yfinance.py` (company name for news queries),
`assistant/provider.py` (`make_provider()`, same `ASSISTANT_MODEL` default as briefs).

## Source details

- **Google News RSS** — `https://news.google.com/rss/search?q=<query>`, keyless.
  Query is `"<company shortName>" OR <TICKER> stock` when the yfinance fundamentals
  snapshot provides a name (avoids single-letter-ticker noise, e.g. F), else
  `<TICKER> stock`. Window: last 14 days, parsed with `feedparser`.
- **Seeking Alpha RSS** — `https://seekingalpha.com/api/sa/combined/<TICKER>.xml`.
  Titles + timestamps only. Most fragile source (Cloudflare moods): polite UA, one
  attempt, on failure tier 2 proceeds on Reddit alone.
- **Reddit** — praw with client-credentials OAuth (`REDDIT_CLIENT_ID`,
  `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`). Per subreddit list:
  `subreddit.search("<TICKER> OR $<TICKER>", time_filter="week", sort="relevance")`,
  title + selftext excerpt, score, num_comments. **Missing credentials is a soft
  degrade, not an error**: tier 2 runs on Seeking Alpha alone, tier 3 on
  Stocktwits + Trends.
- **Stocktwits** — `https://api.stocktwits.com/api/2/streams/symbol/<TICKER>.json`,
  keyless, ~200 req/hr/IP. Last 30 messages: body, timestamp, and the user-tagged
  `entities.sentiment` (Bullish/Bearish/None) → bull/bear ratio computed mechanically.
- **Google Trends** — existing loader; spike ratio = mean(last 7d) / mean(prior 90d),
  reported as e.g. `2.4x baseline`. pytrends 429s are a known mood: degrade silently.
- All fetchers: 8s timeout, no retry in v1 (the page has a Refresh button), errors
  recorded per source in the tier report.

## Pack assembly (`packs.py`)

- Per-source item caps: T1 25 headlines, T2 20 SA + 20 Reddit, T3 20 WSB + 30 ST.
- Each item rendered as one line: `[i] (source, age) text` — text truncated to 280
  chars; total per-tier pack capped at ~10k chars.
- Cross-source dedupe within a tier by normalized title (yfinance news and Google News
  overlap heavily).
- Pack header includes the tier's mechanical metrics (e.g. ST bull/bear ratio) so the
  LLM sees them as context.

## Scoring (`scoring.py`)

One provider call per non-empty tier (3 max per lookup, `claude-haiku-4-5` via
`make_provider()` — pennies), demanding strict JSON:

```json
{
  "score": -1.0..1.0,
  "stance": "bearish|neutral|bullish|mixed",
  "confidence": 0.0..1.0,
  "summary": "2-3 sentences",
  "key_themes": ["..."],
  "evidence_indices": [3, 7, 12]
}
```

- Sanitization mirrors `briefs.py`: clamp numerics, enum-validate stance, cap list
  lengths. On JSON parse failure: keep raw text as `summary`, `score=None`, tier
  status → `degraded`.
- **`evidence_indices` reference pack item numbers**; `report.py` resolves them back
  to real `{title, source, url, age}` rows. The LLM never emits URLs, so evidence
  links cannot be hallucinated. Out-of-range indices are dropped.
- Mechanical metrics are computed in plain code from loader rows, independent of the
  LLM call, and reported alongside.

## Aggregation & report (`report.py`, `types.py`)

- `run_sentiment(ticker, *, refresh=False, provider=None) -> SentimentReport`.
- Source fetches run concurrently (thread pool, sync requests); the up-to-3 tier LLM
  calls also run concurrently. Worst-case wall ≈ max(fetch) + max(LLM) ≈ 15s.
- `TierReport`: tier id/label, status (`ok|degraded|no_data`), score, stance,
  confidence, summary, key_themes, evidence (resolved rows), mechanical metrics dict,
  per-source statuses, item counts.
- `SentimentReport`: ticker, as_of (UTC), status (`ok|partial|no_data`), tier
  reports, `overall_bias` = mean of available tier scores (tiers with `score=None`
  or `no_data` excluded),
  `divergence` = pair + gap when any two tier scores differ by ≥ 0.6 (module constant
  `DIVERGENCE_GAP`), e.g. `{"pair": ["viral", "official"], "gap": 1.0}`.
- All tiers empty → report status `no_data`; the page says so honestly — never a
  fake neutral.

## Caching

- Each loader caches its raw fetch per `(ticker, date)` parquet under the existing
  convention: `data/processed/<source>/<TICKER>/<YYYY-MM-DD>.parquet`
  (sources: `news/google_news`, `forums/seeking_alpha`, `forums/reddit/<subreddit>`,
  `social/stocktwits`). `refresh=True` refetches.
- The finished report is saved to
  `data/processed/sentiment/reports/<TICKER>/<YYYY-MM-DD>.json`. Same-day re-lookups
  render instantly from this JSON unless Refresh is clicked — and the directory
  quietly accrues the forward history the future batch mode needs.

## Webapp

- `GET /sentiment` — page with a ticker input. Submitting calls
  `GET /api/v1/sentiment/{ticker}` (optional `refresh=1`) from JS with a progress
  spinner (~10–20s on a cold fetch), then renders client-side — the planner-page
  pattern; no SSE.
- Render: three tier cards (score dial, stance badge, confidence, mechanical metrics
  row, summary, evidence links, per-source status footnotes) + an overall strip with
  `overall_bias` and the divergence callout when present.
- Nav link added wherever the other pages are cross-linked.
- Endpoint stays sync (FastAPI runs it in its threadpool); errors return a JSON body
  the page renders as a degraded/empty state.

## Dependencies & environment

- `feedparser>=6.0` — new core dependency (RSS parsing; tiny, pure Python).
- `praw>=7.7` — **moves from `ingest-extra` to core deps** (already pinned there;
  tiny, pure Python; keeps a single declaration).
- New env vars `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT`
  added to Render env + the `trading-models-secrets` Modal secret. User action:
  create the app at reddit.com/prefs/apps (script type). Absent creds = degraded
  tiers, never a crash.
- `ANTHROPIC_API_KEY` already deployed for the chat console; reused.

## Testing

Repo convention — no live network in tests, all HTTP mocked with recorded fixtures:

- Loader parsing: Google News RSS XML, SA RSS XML, Reddit submission JSON (praw
  mocked at client level), Stocktwits JSON — including empty and malformed payloads.
- Pack assembly: caps, truncation, dedupe, index stability.
- Scoring sanitization: valid JSON, malformed JSON (degraded path), out-of-range
  clamps, bad evidence indices dropped.
- Aggregation: overall_bias with missing tiers, divergence threshold both sides,
  all-empty → `no_data`.
- Webapp: page renders; API endpoint returns a stubbed-engine report; refresh flag
  plumbed through.

## Non-goals (v1)

- X/Twitter and TikTok ingestion (revisit only if an API budget appears; the viral
  tier is adapter-shaped so an X adapter slots in without redesign).
- Historical backfill / backtestable sentiment series (forward accrual only).
- A composite trading signal fed into models or tickets — this is a research view.
- Assistant tool, CLI script, planner integration, and the nightly cron job itself
  (the engine is cron-ready; wiring the Modal job is a later, separate change).
