# Data sources

How to obtain each data feed used in this repo. The repo prefers **free,
no-key** sources for the seed models so anyone can clone and reproduce.
Paid / keyed sources are documented as upgrades.

## Currently wired in

### yfinance — equities daily bars

- **What**: Adjusted daily OHLCV for stocks, ETFs, indices, and crypto
  pairs.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.equities.yfinance`](../tradinglib/loaders/equities/yfinance.py)
- **Notes**: Unofficial wrapper around Yahoo Finance's public charts API.
  Occasionally breaks when Yahoo changes their endpoint. Rate-limited but
  generous for personal use. Fine for prototyping; not for production.

### yfinance — earnings calendar

- **What**: Per-ticker earnings event dates (`[ticker, earnings_datetime,
  session]`, UTC-aware; `session` is `bmo`/`amc`/`unknown`).
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.events.earnings`](../tradinglib/loaders/events/earnings.py)
- **Notes**: Wraps `yfinance` `Ticker.get_earnings_dates`. Mocked in tests,
  never called live. Cached point-in-time to
  `data/processed/events/earnings/<ticker>/<snapshot>.parquet` (snapshot
  date in the path, so no future leak). Provider is pluggable behind the
  same schema. See [`data/ingestion/events/README.md`](../data/ingestion/events/README.md).

### DoltHub — historical option chains (`post-no-preference/options`)

- **What**: EOD option-chain quotes (bid/ask/IV per contract) for US single
  names, from the community DoltHub database via its free SQL API. Drives the
  earnings-straddle Phase-2 quote-to-quote backtest.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.options.dolthub`](../tradinglib/loaders/options/dolthub.py)
- **Cache**: `data/processed/options/dolthub/<ticker>/<date>.parquet` —
  untracked (`data/processed/` is gitignored) and fully reproducible from the
  API; an empty API result caches an empty frame so the miss is remembered.
- **Query discipline**: every query must filter on exact `date` AND
  `act_symbol` (the table's PK prefix). Anything else scans a ~1e9-row table
  and times out (`context deadline exceeded`, observed live 2026-06-09).
- **Constraints** (all verified empirically against the live API; consumers
  must handle every one):
  - EOD quotes only — no intraday.
  - **Mon/Wed/Fri-only coverage before ~Oct 2024** (Tue/Thu have zero rows
    table-wide in that era); daily coverage after. Consumers must snap to
    covered dates (see the bounded snapping in
    `models/options/03-earnings-straddle-spy/real_chain.py`).
  - Each (date, symbol) lists only **~3 Friday expirations** at tenor-anchored
    slots (~2/4/7 weeks out); the front week is **never** listed, and the
    visibility window rolls day to day.
  - The **~27-strike band is re-sampled daily** around spot; the grid phase can
    shift between days, so a strike present one day can be absent the next.
  - Strikes/quotes are **contemporaneous, never retro-adjusted for splits** —
    consumers pairing them with adjusted closes need an explicit split table
    (see `SPLITS` in `scripts/earnings_straddle_real_chain_backtest.py`).
  - Pre-rename tickers are keyed under the old symbol (META before 2022-06-09
    is `FB`; the loader aliases this).
  - Known holes: 2024-08-01..06 are empty table-wide; TSLA has 2020 gaps.

### yfinance — forward option-chain snapshots

- **What**: Point-in-time snapshots of the live chain (per-strike bid/ask/IV,
  expirations ≤ 45 calendar days out) in the same canonical schema as the
  DoltHub loader plus a `spot` column, so the backtest can consume either
  source once enough forward history accrues. Purpose: this is the only true
  **point-in-time OOS dataset** the earnings straddle will ever have — value
  accrues with calendar time.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.options.yf_chain`](../tradinglib/loaders/options/yf_chain.py);
  runner: [`scripts/collect_chain_snapshots.py`](../scripts/collect_chain_snapshots.py)
  (the model's 9-name watchlist).
- **Cache**: `data/processed/options/yf_snapshots/<ticker>/<snapshot-date>.parquet`,
  dated in **US Eastern** so an evening UTC run does not stamp tomorrow's date.
- **Notes**: Idempotent per (ticker, ET day) — an existing file is never
  re-fetched. Failure semantics (return sentinels): `-1` already snapshotted
  today (skipped), `-2` fetch failed (nothing written — re-run retries), `0`
  no expirations in window (nothing written — re-run retries), `> 0` rows
  written. Suggested cadence: once per trading day in the **16:15–19:59 ET**
  window — after the options close, comfortably inside the same ET calendar
  day — so the snapshot is stamped with the trading day it represents.
  E.g. Windows Task Scheduler:
  `schtasks /create /tn chain-snapshots /tr "uv run python scripts/collect_chain_snapshots.py" /sc daily /st 16:30`.

### Google News RSS — ticker headlines

- **What**: Recent news headlines per ticker (`[ticker, published, title,
  publisher, url]`), query `"<TICKER> stock when:14d"`.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.news.google_news`](../tradinglib/loaders/news/google_news.py)
- **Notes**: Public RSS endpoint; mocked in tests. Snapshot-cached per UTC day.
  Tier-1 source for the `/sentiment` page.

### Seeking Alpha RSS — per-ticker article titles

- **What**: Article/analysis titles per ticker (`[ticker, published, title, url]`).
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.forums.seeking_alpha`](../tradinglib/loaders/forums/seeking_alpha.py)
- **Notes**: Seeking Alpha's public RSS feed (titles only — no bodies, no API).
  The most fragile sentiment source (Cloudflare moods); failures degrade to
  empty and Tier 2 proceeds on Reddit alone.

### Reddit — forum posts mentioning a ticker

- **What**: Posts from configurable subreddits (`[ticker, subreddit, created,
  title, text, score, num_comments, url]`), last week, search `"<T> OR $<T>"`.
- **Cost**: Free (OAuth app).
- **Setup**: Create a **script** app at <https://www.reddit.com/prefs/apps>; set
  `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (and optionally
  `REDDIT_USER_AGENT`). Without credentials the sentiment engine skips Reddit
  sources gracefully.
- **Loader**: [`tradinglib.loaders.forums.reddit`](../tradinglib/loaders/forums/reddit.py)
- **Notes**: praw client, cached per (subreddit, ticker, day) — cache hits need
  no credentials. Serves Tier 2 (serious subs) and Tier 3 (r/wallstreetbets).

### Stocktwits — retail message stream

- **What**: Last ~30 messages per symbol with user-tagged Bullish/Bearish labels
  (`[ticker, created, body, sentiment, username, url]`).
- **Cost**: Free.
- **Setup**: None — no API key required (~200 requests/hour/IP).
- **Loader**: [`tradinglib.loaders.social.stocktwits`](../tradinglib/loaders/social/stocktwits.py)
- **Notes**: The user tags feed the mechanical bull/bear ratio on the
  `/sentiment` page — free ground truth, no LLM involved.

### Bluesky — cashtag post search

- **What**: Top posts mentioning `$<TICKER>` over the last week
  (`[ticker, created, text, handle, likes, reposts, url]`), engagement-ranked.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.social.bluesky`](../tradinglib/loaders/social/bluesky.py)
- **Notes**: Keyless search on `api.bsky.app` (verified 2026-06-11; the
  deliberately-public AppView host gates search, so this endpoint may close
  someday — failures degrade the viral tier gracefully, and an authenticated
  app-password session is the documented upgrade path). Tier-3 source for the
  `/sentiment` page alongside r/wallstreetbets and Stocktwits.

## Planned / not yet wired in

### Polygon.io — higher-quality equities

- **What**: Trades, quotes, aggregates, and reference data for US equities
  + options. Production-grade.
- **Cost**: Free tier (5 calls/min, 2 years of history). Paid plans start
  around $30/mo.
- **Setup**: Sign up at <https://polygon.io>, get an API key, set
  `POLYGON_API_KEY` in `.env`.

### Alpaca — equities + paper trading

- **What**: Free real-time and historical bars for US equities. Includes a
  paper-trading API for executing simulated strategies live.
- **Cost**: Free for the data and paper trading. Live trading is free too
  (commission-free).
- **Setup**: Sign up at <https://alpaca.markets>, generate keys, set
  `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET` in `.env`.

### Binance / Coinbase — crypto

- **What**: Trades, quotes, OHLCV, and full order books for spot + futures
  markets.
- **Cost**: Free for market data — no API key required for read-only
  endpoints.
- **Setup**: None for market data. Provide keys only if you call
  authenticated endpoints (account balance, order history).
- **Notes**: Both exchanges expose WebSocket streams for live order-book
  data. CCXT (<https://github.com/ccxt/ccxt>) is a convenient cross-exchange
  Python wrapper.

### SEC EDGAR — fundamentals

- **What**: Filings (10-K, 10-Q, 8-K, etc.) and structured financial
  statements for all US-listed companies.
- **Cost**: Free.
- **Setup**: None.
- **Notes**: <https://www.sec.gov/edgar.shtml>. The `sec-api` and
  `edgar` Python packages wrap the data. Filings are XBRL-tagged — you can
  pull income statements, balance sheets, and cash-flow statements
  programmatically.

### Reddit — sentiment

- **What**: Posts and comments from subreddits like `r/wallstreetbets`,
  `r/Bitcoin`, `r/CryptoCurrency`.
- **Cost**: Free (with rate limits).
- **Setup**: Create a Reddit app at <https://www.reddit.com/prefs/apps>,
  set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` in
  `.env`. Use [PRAW](https://praw.readthedocs.io/).
- **Caveat**: Historical posts older than ~6 months are hard to backfill
  since Pushshift was restricted. Best used for live signal generation
  going forward.

### NewsAPI — news headlines

- **What**: News headlines + URLs across many publications.
- **Cost**: Free tier limited to 100 requests / day and 1 month of
  history. Paid plans for backfill.
- **Setup**: Sign up at <https://newsapi.org>, set `NEWSAPI_KEY` in `.env`.

### Google Trends — search interest

- **What**: Relative search interest for any query over time. Useful as a
  retail-attention proxy.
- **Cost**: Free, no key.
- **Setup**: Install `pytrends`. Daily resolution available only for
  windows under ~9 months; longer windows return weekly data.
- **Caveat**: `pytrends` is community-maintained and rate-limited.

## How to add a new source

1. Add a loader at `tradinglib/loaders/<asset_class>/<source>.py` that
   downloads, canonicalizes, and writes parquet to
   `data/processed/<source>/...`. Mirror the structure of
   [`yfinance.py`](../tradinglib/loaders/equities/yfinance.py).
2. Document the source in `data/ingestion/<asset_class>/README.md`.
3. If the source needs an API key, add a placeholder to `.env.example`
   and reference it from the loader via `os.environ` or
   `python-dotenv`.
4. Add a row to the table at the top of this file.
5. If a model uses the source, mention it in the model's
   `model.md` frontmatter under `data_sources:`.
