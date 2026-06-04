# Delta-Hedged Long Option on SPY

The options family's hello-world. Holds a long ~1-month ATM SPY call,
delta-hedged to zero each bar with the underlying, rolled at expiry.

## What it tests

Continuous delta-hedging removes first-order directional exposure, so the
residual P&L is the gap between **realized** volatility (how much SPY actually
moved) and the **implied** volatility the option was priced at. It exercises
the whole options stack end-to-end: Black-Scholes pricing & Greeks
(`tradinglib.options.pricing`), the multi-leg position model
(`tradinglib.options.instruments`), the mark-to-market engine
(`tradinglib.backtest.options_engine`), and the GBM Monte Carlo simulation
(`tradinglib.options.simulate`).

## Result

Over 2023-2024 the strategy posts a negative Sharpe: it consistently bleeds
theta because the 18% implied-vol pricing assumption sat above SPY's realized
vol for most of the window. That is the expected behavior for long-vol — and
exactly why the model is useful as a clean, interpretable demonstrator rather
than a money-maker.

## Caveats

- Constant implied-vol assumption — there is no vol surface or term structure
  in phase 1. The historical-chain loader (future phase) will replace this.
- European exercise. The American CRR pricer exists and is tested, but the
  clean vol story uses European pricing.
- Daily rehedge only; intraday gamma P&L is not captured.

## Reproduce

```bash
uv run python models/options/01-delta-hedged-long-option-spy/backtest.py
```
