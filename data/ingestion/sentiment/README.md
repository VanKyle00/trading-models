# Sentiment / alt-data ingestion

## Sources

| Source | Type | Loader | API key | Notes |
| --- | --- | --- | --- | --- |
| Google Trends (via pytrends) | search interest, weekly/monthly | [`tradinglib.loaders.sentiment.google_trends`](../../../tradinglib/loaders/sentiment/google_trends.py) | none | free; community-maintained wrapper; occasional rate limits |

## Google Trends

Relative search interest (0-100) for a query over a chosen window.
Resolution depends on window length: daily under ~9 months, weekly under
5 years, monthly above.

```python
from tradinglib.loaders.sentiment.google_trends import load_interest

# Weekly resolution for a 4-year window
series = load_interest("bitcoin", timeframe="2021-01-01 2024-12-31")
```

The first call hits Google Trends and caches the response to
`data/processed/google_trends/<cache_key>/interest.parquet`. Subsequent
calls read from the cache. Values are *relative within the query window*,
so changing the window produces a different scale — query once over the
full window of interest.

## Planned

- **Reddit** via PRAW — see `.env.example` for keys. Historical posts
  > ~6 months are hard to backfill since Pushshift was restricted.
- **NewsAPI** — headlines with sentiment scoring (e.g., VADER, FinBERT).
- **HackerNews** — Algolia's free search API.
