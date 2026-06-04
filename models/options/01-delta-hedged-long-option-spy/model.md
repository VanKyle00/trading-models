---
name: Delta-Hedged Long Option on SPY
family: options
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars]
tickers: any
default_ticker: SPY
status: working
sharpe_oos: -6.94
max_drawdown: -0.08
---

Buy a ~1-month ATM call on SPY and delta-hedge to zero every bar, rolling a
fresh option at expiry. Continuous hedging strips out directional exposure so
the P&L isolates realized-vs-implied volatility. Doubles as the options
pipeline's end-to-end smoke test: pricing, Greeks, expiry-roll, mark-to-market
accounting, and a Monte Carlo outcome distribution. See [`README.md`](README.md).
