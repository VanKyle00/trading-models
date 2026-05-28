# Equities ingestion

## Sources

| Source | Type | Loader | API key | Notes |
| --- | --- | --- | --- | --- |
| yfinance | daily OHLCV | [`tradinglib.loaders.equities.yfinance`](../../../tradinglib/loaders/equities/yfinance.py) | none | unofficial, rate-limited, free; good for prototyping |

## yfinance

Adjusted daily OHLCV for stocks, ETFs, indices, and a wide set of crypto
pairs. Returned by Yahoo Finance's public charts API; the `yfinance`
package wraps it. Unofficial, no SLA, occasionally breaks when Yahoo
changes their endpoint — fine for portfolio work, not for production
trading.

Usage:

```python
from tradinglib.loaders.equities.yfinance import load_daily

df = load_daily("SPY")                  # full history, cached to parquet
df = load_daily("SPY", start="2015-01-01", end="2024-12-31")
df = load_daily("SPY", refresh=True)    # force redownload
```

The first call downloads the full history and caches it to
`data/processed/yfinance/SPY/daily.parquet`. Subsequent calls read the cache
and filter to the requested date range in-memory.
