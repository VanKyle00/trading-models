# Crypto ingestion

## Sources

| Source | Type | Loader | API key | Notes |
| --- | --- | --- | --- | --- |
| Binance (public REST) | daily OHLCV klines | [`tradinglib.loaders.crypto.binance`](../../../tradinglib/loaders/crypto/binance.py) | none | free; rate-limited to 1200 weight/min/IP |

## Binance

Daily candles for any spot pair listed on Binance, pulled from
``api.binance.com/api/v3/klines``. No API key needed for read-only
market data. Pagination is handled by the loader — Binance caps each
call at 1000 candles, so multi-year history requires several requests
(handled transparently).

```python
from tradinglib.loaders.crypto.binance import load_daily

df = load_daily("BTCUSDT")                                # full history
df = load_daily("BTCUSDT", start="2020-01-01", end="2024-12-31")
df = load_daily("BTCUSDT", refresh=True)                  # force redownload
```

The first call paginates through the API and caches the result to
`data/processed/binance/<symbol>/daily.parquet`. Subsequent calls read
the parquet.

Common symbols to try: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`. The
quote asset is appended directly — Binance does not use a separator.

## Planned

- **Trade-level (tick) data** for one symbol via the WebSocket
  ``@aggTrade`` stream. Required for any microstructure model.
- **Level-2 order book** via the partial book depth WebSocket. Required
  for orderbook-imbalance / microprice features.
- **Coinbase Advanced Trade** as a cross-venue check.
