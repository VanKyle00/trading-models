---
name: XGBoost Next-Day Return on SPY
family: ml
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars]
tickers: [SPY]
# Model is evaluated only on its held-out OOS slice; earlier dates have no
# data to score, so bound the picker to the OOS window.
date_min: 2022-01-14
date_max: 2024-12-31
params: []
status: working
sharpe_oos: 0.96
max_drawdown: -0.12
---

Gradient-boosted regressor predicting next-day SPY log return from a small
set of technical features. 80/20 chronological train/test split; metrics are
reported on the held-out OOS window. See [`README.md`](README.md) for the
full writeup.
