# Backtest methodology

Every model in this repo runs through the same backtest engine
([`tradinglib.backtest`](../tradinglib/backtest/engine.py)) under the same
assumptions, so cross-model comparisons are meaningful. This document
captures those assumptions.

## Execution model

- **Bar alignment**: A signal computed using information available at the
  *close* of bar `t` is treated as a position taken *starting at bar t+1*.
  Mechanically, the engine lags the signal series by one bar before
  multiplying by per-bar returns. This is the single most important
  guardrail against look-ahead bias.
- **Fill price**: `run_backtest` fills at the **next bar's open by default**
  (`fill="next_open"`), which requires an `open_prices` series. The entry bar
  earns `open → close`; held bars stay close-to-close. This removes the
  optimism of filling at the very close used to make the decision. Pass
  `fill="decision_close"` for close-to-close fills. The legacy
  `execution_prices=` argument is a deprecated alias for `open_prices`.
- **Position units**: Positions are expressed as a fraction of current
  equity. A signal of `1.0` means "be fully invested"; `-1.0` means "be
  fully short"; `0.5` means "deploy half of equity long".
- **Compounding**: PnL compounds — the equity curve is
  `cumprod(1 + net_returns) * initial_capital`.
- **Single-asset**: The v1 engine is single-asset and assumes the user's
  signal already reflects any sizing logic. A multi-asset / portfolio
  variant is a future addition.
- **Two front-ends, one math core.** `run_backtest` is vectorized — pass
  a price series and a signal series. `run_event_backtest` is event-
  driven — pass a sequence of `Bar` events and a `Strategy` callback.
  Both produce identical `BacktestResult` shapes and identical PnL,
  because the event engine just records the strategy's per-bar target
  and delegates the math to the vectorized engine. Choose the event
  engine when the strategy is path-dependent (stop-losses, trailing
  stops, regime filters); choose the vectorized engine for pure-signal
  strategies.

## Options (mark-to-market) results

Options strategies run through `tradinglib.backtest.options_engine`, which
marks a multi-leg position to market each bar rather than using the linear
`position × return` math. The resulting `BacktestResult` reuses
`compute_metrics`, but two fields are reinterpreted:

- **`position`** — net portfolio *delta* expressed as a fraction of equity
  (`net_delta_shares × spot / equity`), not a target weight.
- **`turnover`** — traded notional (underlying + option premium) divided by
  equity for that bar.

`equity_curve` is the portfolio's mark-to-market value and `returns` is its
bar-over-bar percent change, so Sharpe/Sortino/drawdown stay comparable to
every other model.

## Transaction costs

- **Linear in turnover**: Cost per bar = `turnover * (fee_bps + slippage_bps) / 10_000`,
  where turnover is `|position_t - position_{t-1}|`.
- **Defaults**: 1 bp commission + 0.5 bp slippage = 1.5 bp round-trip per
  unit of turnover. These match liquid-name US-equity conditions on
  retail-broker pricing in 2025. Crypto markets are similar on top venues;
  small-cap equities should use 5–20 bps.
- **No spread / market impact modeling**: The cost model does not separately
  size bid/ask spread or impact. Treat the slippage parameter as a
  catch-all for "everything that erodes the price you wanted vs the price
  you got".

## Metrics

All metrics live in
[`tradinglib.backtest.metrics`](../tradinglib/backtest/metrics.py) and are
JSON-serialized to each model's `results/metrics.json`.

- **Sharpe ratio**: `sqrt(periods_per_year) * mean(returns) / std(returns)`.
  Risk-free rate is zero. The default `periods_per_year=252` matches daily
  US-equity bars; minute bars should pass `252 * 6.5 * 60 = 98_280`.
- **Sortino ratio**: Like Sharpe but the denominator is the std of negative
  returns only — credits the strategy for upside volatility.
- **Annualized return**: Compounding the realized returns to a one-year
  horizon: `total_growth ** (periods_per_year / n_bars) - 1`.
- **Maximum drawdown**: Largest peak-to-trough decline in the equity
  curve, expressed as a negative ratio.
- **Hit rate**: Fraction of *active* bars (returns ≠ 0) that were
  positive. Reported but not used to draw conclusions — high hit rates are
  routinely associated with strategies that lose money on a few big losers.
- **Probabilistic Sharpe Ratio (PSR)**: Probability that the true (non-
  annualized) Sharpe is greater than zero, given the track length and the
  skew and kurtosis of the return series. Fat tails and short tracks lower it.
- **Deflated Sharpe Ratio (DSR)**: PSR with the benchmark raised to the
  *expected maximum* Sharpe across `n_trials` independent configurations
  (Bailey & López de Prado, 2014). It corrects for selection bias from trying
  many strategies. `n_trials` defaults to 1 for direct single-config backtests
  (DSR then equals PSR), but the walk-forward demos perform a parameter search
  and pass the true grid size as `n_trials` (9 for SMA, 4 for XGBoost), so
  their Deflated Sharpe is genuinely deflated below the Probabilistic Sharpe.

### Tournament metrics conventions

The nightly per-ticker tournament reuses `compute_metrics` unchanged; these are
the conventions and known approximations behind its numbers.

- **Flat bars are included in Sharpe.** The stitched out-of-sample Sharpe is
  computed over every OOS bar, including bars where the strategy is flat.
  Low-activity rules are diluted toward zero — intended: it is the Sharpe of
  *running this rule on this ticker*, not of cherry-picked active periods.
- **Drawdown baseline.** The engine's one-bar signal lag forces position 0 on
  the first bar, so equity curves start exactly at initial capital and a
  strategy that loses from its first active bar reports that loss as drawdown.
- **Deflated Sharpe approximation.** The expected-maximum-Sharpe benchmark uses
  the candidate's own estimator variance (Lo 2002) rather than the cross-trial
  variance of all trial Sharpes (Bailey & López de Prado 2014); under the null
  hypothesis the two coincide, and the estimator variance is available without
  persisting every trial's stitched return series.
- **Survivorship bias.** Tournaments test *today's* index constituents on ~3.3
  years of history (a 1,200 *calendar*-day lookback); names that fell out of
  the index are absent, which flatters
  trend/breakout results. Until a point-in-time universe exists, treat absolute
  metric levels with suspicion; comparisons between strategies on the same
  ticker are unaffected.
- **Costs.** 1 bp commission + 0.5 bp slippage per unit turnover, linear.
  Defensible for liquid Russell 1000 names filled at the next open; optimistic
  for less liquid names or stop-driven fills.
- **1 − DSR as a pseudo-p-value.** The nightly Benjamini-Hochberg pass treats
  `1 − DSR` as a p-value for "no edge" across every ticker-stance tournament
  run that night (`fdr_alpha = 0.10`, matching the survival bar). It is a
  Gaussian-approximation probability, not a textbook p-value; it orders and
  tiers candidates — survivors failing it demote to the watchlist — and is
  not a published significance claim.

## Train / test discipline

- **Chronological split only** — never shuffle time-series data before
  splitting. Future data must never appear in the training window.
- **Default split**: Each ML model uses a chronological 80/20 train/test
  unless walk-forward validation is specified. Walk-forward (rolling or
  expanding window) is the right move once a model proves itself on a
  fixed split.
- **OOS reporting**: When a model has a train/test split, the metrics in
  `MODELS.md` and `model.md` are the *out-of-sample* numbers. The
  `train_metrics.json` (if present) carries the in-sample stats.

## What this engine does *not* model

- **Slippage by order size** — A real $1M order moves the market
  differently than a $1k order. The repo treats slippage as a flat
  per-unit-turnover cost.
- **Borrow costs** for short positions — Real shorts pay an annualized
  borrow fee, which can be material for hard-to-borrow names.
- **Funding rates** in crypto perpetuals — Funding is paid every few hours
  on perp positions and can dominate returns for high-frequency strategies.
- **Margin / leverage costs** — Real brokers charge interest on margin
  loans. We assume positions are fully funded by equity.
- **Tax** — Out of scope.

Models that depend critically on one of these (e.g., a basis-trade strategy
on perp funding) should add their own cost line and document it in their
README.

## Common pitfalls and how the engine helps

| Pitfall | What the engine does |
| --- | --- |
| Using same-bar info as the signal | Lags every signal by one bar |
| Inconsistent metrics across models | Single `compute_metrics` function applied to every result |
| Forgetting to pay transaction costs | Costs default to non-zero (1 bp + 0.5 bp) |
| Mixing train and test data | Engine doesn't enforce, but every ML seed model uses a chronological split — copy that pattern |
| Equity index drift across models | Every `BacktestResult` shares the same `BacktestResult` shape, including `config` recording the parameters used |

## Walk-forward validation & the next-open default

`run_backtest` now defaults to `fill="next_open"` and requires an `open_prices`
series; pass `fill="decision_close"` for the prior close-to-close behavior. The
legacy `execution_prices=` argument is a deprecated alias.

The `tradinglib.validation` package adds a walk-forward harness
(`walk_forward`), grid search with an honest trial count (`grid_search`), and
sensitivity / regime diagnostics (`parameter_sensitivity`, `metrics_by_regime`).
A model adopts it by writing one `make_signal(train, test, params)` adapter; the
harness re-optimizes parameters per window and deflates the out-of-sample Sharpe
by the grid size. See `models/classical/01-sma-crossover-spy/walk_forward.py`
and `models/ml/01-gbm-next-day-return-spy/walk_forward.py`.

## Realistic options frictions — synthetic vol surface & spread

The options engine prices and fills through a `VolSurface` and a `SpreadModel`
(`tradinglib/options/surface.py`, `spread.py`) instead of a single constant vol.
`realistic_surface(prices)` anchors ATM implied vol to the underlying's trailing
realized vol (× a volatility-risk premium) and overlays parametric skew and term
structure; `ParametricSpread` fills option legs by crossing a bid/ask that widens
for out-of-the-money and short-dated contracts. The legacy `vol=` argument is a
deprecated alias for `surface=FlatSurface(vol)`.

This is a **stress / plausibility model, not a market-calibrated one**: it tests
whether an edge survives realistic-shaped vol regimes and frictions, not the exact
historical P&L of a specific contract (which needs real options-chain data). See
`models/options/02-directional-call-spy/backtest.py` for a frictionless-vs-realistic
comparison.
