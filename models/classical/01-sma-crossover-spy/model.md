---
name: SMA Crossover on SPY
family: classical
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars]
tickers: any
default_ticker: SPY
status: working
sharpe_oos: 0.75
max_drawdown: -0.34
---

The obligatory hello-world. Long SPY when the 50-day simple moving average is
above the 200-day SMA; otherwise flat. No shorting. Tests the full pipeline
end-to-end. See [`README.md`](README.md) for the writeup.
