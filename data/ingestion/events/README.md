# Events ingestion

## Sources

| Source | Type | Loader | API key | Notes |
| --- | --- | --- | --- | --- |
| yfinance | earnings calendar | [`tradinglib.loaders.events.earnings`](../../../tradinglib/loaders/events/earnings.py) | none | unofficial, rate-limited, free; mocked in tests, never called live |

## yfinance — earnings calendar

Per-ticker earnings event dates from Yahoo Finance, wrapped by the
`yfinance` package's `Ticker.get_earnings_dates`. Unofficial, no SLA,
occasionally breaks when Yahoo changes their endpoint — fine for
prototyping the earnings-straddle pipeline, not for production.

The loader canonicalizes to a DataFrame with columns
`[ticker, earnings_datetime, session]` where `earnings_datetime` is
UTC-aware and `session` is one of `{"bmo", "amc", "unknown"}` (before
market open / after market close / unknown, inferred from the event's
Eastern-time clock).

Usage:

```python
from tradinglib.loaders.events.earnings import get_earnings_dates

df = get_earnings_dates(["AAPL", "MSFT"], start="2023-01-01", end="2024-12-31")
```

### Point-in-time caching

Each fetch is cached to
`data/processed/events/earnings/<ticker>/<snapshot>.parquet`, where
`<snapshot>` is the UTC date the data was pulled. The snapshot date is
part of the path — it never future-leaks into the data — so a backtest can
read the calendar exactly as it was known on a given day.

### Pluggable provider

yfinance is the default provider. The loader is structured so a different
calendar source (a paid feed, a snapshotted forward collector) can be
swapped in behind the same `[ticker, earnings_datetime, session]` schema
without touching the model's signal, sizing, or validation code.
