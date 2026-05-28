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
- **Position units**: Positions are expressed as a fraction of current
  equity. A signal of `1.0` means "be fully invested"; `-1.0` means "be
  fully short"; `0.5` means "deploy half of equity long".
- **Compounding**: PnL compounds — the equity curve is
  `cumprod(1 + net_returns) * initial_capital`.
- **Single-asset**: The v1 engine is single-asset and assumes the user's
  signal already reflects any sizing logic. A multi-asset / portfolio
  variant is a future addition.

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
