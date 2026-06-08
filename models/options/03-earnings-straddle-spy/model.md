---
name: Earnings Event-Vol Straddle on SPY
family: options
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars, yfinance_earnings_calendar]
tickers: any
default_ticker: SPY
supports_costs: false
supports_sizing: false
params:
  - {name: k, label: Edge margin (expected/implied), type: float, default: 1.2, min: 1.01, max: 3.0}
  - {name: lookback, label: Earnings lookback (events), type: int, default: 8, min: 2, max: 20}
  - {name: entry_lead, label: Entry lead (trading days), type: int, default: 3, min: 1, max: 10}
  - {name: exit_offset, label: Exit offset (trading days), type: int, default: 1, min: 0, max: 5}
  - {name: pre_iv, label: Pre-earnings IV (synthetic), type: float, default: 0.45, min: 0.15, max: 1.50}
  - {name: post_iv, label: Post-earnings IV (synthetic crush, must be < pre_iv), type: float, default: 0.25, min: 0.05, max: 0.90}
status: working
sharpe_oos: 0.0
max_drawdown: 0.0
---

Long ATM straddle entered into earnings on a liquid optionable name. The edge is
a selection filter — enter only when the forecast realized move exceeds the
implied move priced into the straddle by a margin k (>1). The straddle is the
expression; the filter is the alpha. Phase 1 is synthetic (an elevated
pre-earnings IV plus a parameterized post-earnings crush, with the constraint
`post_iv < pre_iv` so the synthetic premium is conservative, never tunable to a
fake IV expansion) and clearly not yet tradeable; the realized move comes from
real yfinance bars and expected move from PRIOR earnings events only. The
validation report distinguishes filtered vs unfiltered P&L and applies a
non-parametric bootstrap test, Benjamini-Hochberg FDR across the watchlist, and
trade-level metrics. Walk-forward across earnings seasons is deferred (the
existing harness is built on the vectorized engine; an options-aware
walk-forward is a separate cycle).
