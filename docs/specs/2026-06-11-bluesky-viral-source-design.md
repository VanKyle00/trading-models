# Bluesky as a Tier-3 Viral Source — Design Spec

**Date:** 2026-06-11
**Status:** Approved (pending written-spec review)
**Scope:** Add Bluesky post search as a fourth source in the ticker-sentiment
tool's viral tier, through the adapter seam the original design left for an
X-style source. One new loader + tier-3 wiring + tests + docs. No webapp
changes. Extends `docs/specs/2026-06-11-ticker-sentiment-design.md`.

## Motivation

X is the natural viral-tier source but has no legitimate free read access
(free API tier is write-only; scraping is ToS-hostile and blocked from cloud
IPs). Bluesky is the one X-shaped network with a free, official,
search-capable API, and enough migrated fintwit that cashtag queries return
real volume on liquid names. Adding it gives tier 3 a second message-stream
voice alongside Stocktwits and recovers some of the early-detection lead time
the tool otherwise only catches as an echo (news writeups, Trends spikes).

## Decisions (locked during brainstorming, 2026-06-11)

- **Keyless search on `api.bsky.app` (approach A).** Verified empirically
  2026-06-11: `GET https://api.bsky.app/xrpc/app.bsky.feed.searchPosts`
  returns 200 without auth; `q=$NVDA, sort=top, since=<7d ago>, lang=en,
  limit=25` returned a full page with all needed fields. The deliberately
  public host (`public.api.bsky.app`) 403s `searchPosts` specifically, so the
  endpoint could be gated someday — that risk lands in the tool's standard
  degrade-gracefully class (tier 3 proceeds on Stocktwits + Trends; the
  authenticated app-password session is the documented upgrade path if the
  keyless endpoint closes). No new dependency, no secrets.
- **Cashtag query, top-of-week.** `q=$<TICKER>` (precision over recall; a
  quiet small cap honestly returning nothing beats keyword noise),
  `sort=top` within a 7-day `since` window, `lang=en`. Engagement-ranked is
  the right read for a *virality* tier.
- **Authenticated path (approach B) and curated-handles feeds (approach C)
  rejected** — B is secrets + session code for a hypothetical lockdown
  (YAGNI; documented as fallback), C measures a watchlist, not virality.

## Loader — `tradinglib/loaders/social/bluesky.py`

- `get_bluesky_posts(ticker, *, max_items=25, refresh=False) -> pd.DataFrame`
- Schema (canonical): `[ticker, created, text, handle, likes, reposts, url]`,
  UTC-aware, newest-engagement-first as returned by `sort=top` (no local
  re-sort — the API's ranking IS the signal), capped at `max_items`.
- Request: `https://api.bsky.app/xrpc/app.bsky.feed.searchPosts` with
  `q=$<TICKER>`, `sort=top`, `since=<now-7d, ISO Z>`, `lang=en`,
  `limit=max_items`, polite UA (same string as the Seeking Alpha loader),
  8s timeout, `follow_redirects`. Catch-all → warn + typed empty (house
  loader convention).
- Field mapping per post: `record.text` → text, `record.createdAt` → created
  (→ `datetime64[ms, UTC]` astype with the house ms comment), `author.handle`
  → handle, `likeCount`/`repostCount` → likes/reposts (int, 0 when missing),
  url = `https://bsky.app/profile/<handle>/post/<rkey>` where rkey is the
  final segment of the `at://` uri (empty url if uri/handle missing). Posts
  without text are skipped. NaN-safety per the pandas-3 house rules.
- Snapshot cache: `data/processed/social/bluesky/<TICKER>/<YYYY-MM-DD>.parquet`,
  `refresh=True` bypasses.

## Engine wiring (tier 3 only)

- `report.py`: new `"bluesky"` job in `_fetch_sources` (max_items=25); the
  viral tier's items/metrics/source-status wiring gains the frame; viral
  `source_status` subset becomes `(wsb, stocktwits, bluesky, google_trends)`.
- `packs.viral_items(wsb, stocktwits, bluesky)`: Bluesky items use source
  label `Bluesky @<handle> (+<likes>, <reposts>r)` (mirrors the Reddit
  label), title = text[:80], text = full post text, url, published = created.
- `scoring.viral_metrics(..., bluesky, interest)`: gains
  `bsky_mentions: len(bluesky)`. The viral `_TIER_DESC` mentions Bluesky.
- A Bluesky failure degrades the tier like any content source (it is pack
  content, unlike Trends).

## Testing

- `tests/test_bluesky_loader.py` mirroring the Stocktwits suite: schema +
  field mapping + url building + query params asserted; http error → typed
  empty; snapshot cached (1 HTTP call); max_items cap. Fixture
  `tests/fixtures/sentiment/bluesky.json` is UTF-8 **with emoji in post
  text** (locks the encoding handling the live probe surfaced).
- Updates: `test_sentiment_packs.py` (viral_items signature + a Bluesky item
  assertion), `test_sentiment_scoring.py` (viral_metrics signature +
  bsky_mentions), `test_sentiment_report.py` (stubbed fixture gains
  get_bluesky_posts; full-report assertions extended; a bluesky-error
  degradation case).

## Docs

- `docs/data-sources.md`: Bluesky entry (What/Cost/Setup/Loader/Notes — note
  the keyless-host caveat and app-password upgrade path).
- `README.md`: tier-3 row in the sentiment table gains "Bluesky cashtag
  search"; mechanical-metrics cell gains the mention count.

## Non-goals

- Authenticated Bluesky sessions, firehose/jetstream ingestion, or per-handle
  watchlists.
- Any X integration (revisit only with an API budget).
- Webapp changes (tier cards render new metrics/statuses generically).
