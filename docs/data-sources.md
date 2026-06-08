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
