---
name: Order Flow Imbalance on BTC
family: microstructure
window: intraday
assets: [crypto]
data_sources: [binance_aggTrades]
tickers: [BTCUSDT]
# Ships only the 2024-08-05-crash aggTrades window; other days aren't cached and
# pulling them (~50-100 MB/day) risks OOM on the hosted demo.
date_min: 2024-08-04
date_max: 2024-08-06
supports_costs: false
supports_sizing: false
params:
  - {name: bar_seconds, label: Bar size (seconds), type: int, default: 60, min: 10, max: 300}
  - {name: smooth_window, label: OFI smoothing window, type: int, default: 5, min: 2, max: 30}
  - {name: entry_threshold, label: Entry threshold, type: float, default: 0.20, min: 0.01, max: 1.0}
status: negative-result
sharpe_oos: -86.37
max_drawdown: -0.36
---

Rolling order-flow imbalance on 1-minute BTCUSDT bars, traded via the
event-driven backtest engine. Three-day window covering the 2024-08-05
crash (pre-crash, crash, recovery). The trade-flow continuation
hypothesis is rejected by the data over this period — hit rate 29.7%
and a -36% drawdown. See [`README.md`](README.md) for the writeup and
the discussion of why the bar-frequency-annualized Sharpe shouldn't be
compared apples-to-apples with the daily-bar models in this repo.
