---
name: Order Flow Imbalance on BTC
family: microstructure
window: intraday
assets: [crypto]
data_sources: [binance_aggTrades]
status: negative-result
sharpe_oos: -86.5
max_drawdown: -0.36
---

Rolling order-flow imbalance on 1-minute BTCUSDT bars, traded via the
event-driven backtest engine. Three-day window covering the 2024-08-05
crash (pre-crash, crash, recovery). The trade-flow continuation
hypothesis is rejected by the data over this period — hit rate 29.7%
and a -36% drawdown. See [`README.md`](README.md) for the writeup and
the discussion of why the bar-frequency-annualized Sharpe shouldn't be
compared apples-to-apples with the daily-bar models in this repo.
