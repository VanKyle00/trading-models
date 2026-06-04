---
name: Google Trends Contrarian on BTC
family: alt-data
window: swing
assets: [crypto]
data_sources: [google_trends_bitcoin, yfinance_daily_bars]
tickers: [BTC-USD]
params:
  - {name: window, label: Z-score window (weeks), type: int, default: 4, min: 2, max: 12}
status: negative-result
sharpe_oos: -0.30
max_drawdown: -0.80
---

Contrarian attention model: short BTC when 'bitcoin' search interest is
unusually high (4-week z-score > 1), long when unusually low (z < -1),
flat otherwise. Weekly resolution. See [`README.md`](README.md) for the
full writeup.
