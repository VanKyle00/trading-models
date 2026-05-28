# 01 — SMA Crossover on SPY

**Hypothesis.** When SPY's 50-day simple moving average crosses above its
200-day SMA (the "golden cross" regime), the market is in an uptrend and
being long earns positive return. When the fast SMA falls below the slow
SMA ("death cross"), step aside.

**Data.** SPY daily adjusted bars from yfinance, `2010-01-01` to `2024-12-31`.

**Signal.**

- `fast > slow` → fully long (`position = 1.0`)
- `fast ≤ slow` → flat (`position = 0.0`)
- No shorting in this version.

**Costs.** 1 bp commission + 0.5 bp slippage per unit of turnover.

**Why this model exists.** Hello-world for the repo. A naive SMA crossover
on SPY is well known *not* to beat buy-and-hold after costs over extended
periods. The point here is to exercise the pipeline end-to-end — data
load → signal → backtest → metrics → plot — and verify the infrastructure
works. Every later model can copy this skeleton.

## Reproduce

```bash
uv run python models/classical/01-sma-crossover-spy/backtest.py
```

The first run downloads SPY (cached to `data/processed/yfinance/SPY/`),
computes the signal, runs the backtest, and writes
[`results/metrics.json`](results/metrics.json) and
[`results/equity_curve.png`](results/equity_curve.png).

## Results

See [`results/metrics.json`](results/metrics.json) for the full set
(Sharpe, Sortino, annualized return, max drawdown, hit rate, turnover).
The equity-curve plot compares the strategy against a passive buy-and-hold
benchmark over the same window.

## Takeaways

_To be filled in after the first run._
