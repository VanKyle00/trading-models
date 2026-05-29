---
name: XGBoost Next-Day Return on SPY
family: ml
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars]
tickers: [SPY]
status: working
sharpe_oos: 0.84
max_drawdown: -0.14
---

Gradient-boosted regressor predicting next-day SPY log return from a small
set of technical features. 80/20 chronological train/test split; metrics are
reported on the held-out OOS window. See [`README.md`](README.md) for the
full writeup.
