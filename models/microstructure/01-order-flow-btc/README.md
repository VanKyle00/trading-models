# 01 — Order Flow Imbalance on BTCUSDT

**Hypothesis.** Aggressive buying — taker market-orders lifting the ask —
produces short-term upward pressure on price; aggressive selling pushes
price down. Smoothed across a few minutes, this trade-flow asymmetry is
the canonical microstructure effect documented since Hasbrouck (1991).
If we measure it bar-by-bar and trade with the flow, we should capture
some of that continuation, minus costs.

**Data.** Three days of BTCUSDT aggregated trades from Binance's
public CDN (`data.binance.vision`), bracketing the **2024-08-05 crash**
(pre-crash + crash + recovery). The crash window stress-tests the
model under heavy, symmetric flow.

| Source | Window | Rows | Bars |
| --- | --- | --- | --- |
| `binance_aggTrades` | 2024-08-04 to 2024-08-06 (3 days) | ~8.7M trades | 4,320 1-minute bars |

The loader caches the canonicalized trades to
`data/processed/binance/BTCUSDT/aggTrades/2024-08-05.parquet`; subsequent
runs read the parquet rather than re-downloading the ZIP.

**Feature.** Per minute,

```
OFI = (buy_volume − sell_volume) / total_volume
```

where the side of a trade is the side of the **aggressor**: a trade
flagged `is_buyer_maker=False` is an aggressive buy (`+`); the opposite
flag means aggressive sell (`−`).

The raw OFI is volatile bar-to-bar, so we smooth with a **5-minute
trailing average**.

**Signal.**

- smoothed OFI > +0.20 → fully long (`position = 1.0`)
- smoothed OFI < −0.20 → fully short (`position = -1.0`)
- otherwise → flat (`position = 0.0`)

**Costs.** 2 bp commission + 5 bp slippage = 7 bp round-trip per unit of
turnover, calibrated to retail crypto-exchange execution at small size.

**Engine.** Event-driven (`tradinglib.backtest.run_event_backtest`). The
strategy maintains its own 5-element rolling buffer of OFI values; this
is exactly the kind of stateful per-bar logic the event engine exists for.
The vectorized engine could do the same math via `pandas.rolling`, but
the imperative form is closer to how a tick-level live system would
actually be written.

## Reproduce

```bash
uv run python models/microstructure/01-order-flow-btc/backtest.py
```

First run downloads ~67 MB of trades for the chosen day, parses, caches
the trades to parquet, and runs the backtest. Subsequent runs read the
parquet (much faster). Writes [`results/metrics.json`](results/metrics.json)
and [`results/equity_curve.png`](results/equity_curve.png).

To run a longer window, edit `START` / `END` in `backtest.py`. The
loader handles multi-day stitching transparently — each day adds one
HTTP request and one parquet to the cache.

## Results — negative

Over the 3-day window covering the 2024-08-05 crash:

| Metric | Value |
| --- | --- |
| Total return (3 days) | **-36%** |
| Max drawdown | **-36%** |
| Hit rate | **29.7%** |
| Bars (1-minute) | 4,320 |
| "Annualized" Sharpe (`periods_per_year=525_600`) | -86.5 |

**The trade-flow continuation hypothesis is rejected by this sample.**
Hit rate 29.7% means the position is wrong more than twice as often as
right. The strategy bleeds money consistently across the pre-crash,
crash, and recovery regimes — it isn't a single bad-day artefact.

The likely interpretation: on heavily-traded venues like Binance,
aggressive flow is rapidly absorbed by quoting/market-making activity.
By the time a 5-minute average of OFI has crossed ±0.2, the price has
already moved through the level where entering with the flow makes
sense; the next minute is more likely to mean-revert than continue.
This is consistent with the academic literature on price-impact decay
within seconds-to-minutes timescales.

**On the comparison-with-other-models question.** The Sharpe of -86.5
is annualized assuming 525,600 minute bars per year, which makes the
magnitude visually very different from the daily-bar models in this
repo (where annualization uses 252). The *direction* (negative,
clearly losing money) and the scale-invariant metrics (hit rate, max
drawdown) are the relevant comparisons. A "daily-equivalent Sharpe"
would be ``-86.5 / sqrt(525_600/252) ≈ -1.9`` — still very negative
but on the same scale as the daily-bar models.

## Why ship a losing model?

Same reason as the [Google Trends contrarian model](../../alt-data/01-google-trends-btc/):
a portfolio that only shows winners is usually overfit. Shipping the
microstructure pipeline with a real negative result demonstrates that
the infrastructure works end-to-end — tick-data ingestion (5.3M
trades / day), per-bar microstructure feature extraction, event-driven
backtest, standardized metrics — without manufacturing a positive
result through parameter search.

## Caveats

- **Single-day sample.** One day = 1440 1-minute bars is enough to
  produce an equity curve but the per-day Sharpe is too noisy to draw
  firm conclusions. The right next step is to run this over 30+ days
  and report the cross-sectional distribution of daily Sharpes.
- **Trade-flow signed labelling is a proxy.** The Binance `is_buyer_maker`
  flag is reliable, but only the *aggressor's* side is observable —
  there is no information here about resting limit-order book changes.
  A higher-quality OFI uses L2 depth-update events directly; that's a
  follow-up once L2 ingestion is wired in.
- **Crash-day choice.** Picking 2024-08-05 stress-tests the strategy
  but also biases the result toward strategies that handle large
  symmetric flow. Repeating on a calm day (e.g., 2024-07-15) is the
  obvious robustness check.
- **No latency model.** The backtest assumes the signal computed at the
  close of bar `t` is executed at the close of bar `t+1`. Real execution
  inside a 1-minute bar would face spread / slippage proportional to
  the urgency of the order; the 7 bp cost charge is a rough
  approximation.

## Takeaways

- The simple trade-flow continuation rule does not work on a high-volume
  spot venue at minute frequency. Aggressive flow gets absorbed by
  quoting before a 5-minute average crosses the entry threshold.
- The infrastructure (tick loader + microstructure feature library +
  event engine) works end-to-end and is the real deliverable here. The
  next experiment to run is either: (a) faster reaction (raw OFI rather
  than 5-minute average, possibly with smaller threshold), (b) a
  contrarian flip of the rule on the same data, or (c) using L2-derived
  OFI rather than trade-side OFI (would require depth-update ingestion).
- Repeating this across many days and reporting the distribution of
  daily PnL is the right way to draw firmer conclusions; 3 days is too
  few to claim the strategy "doesn't work" in general — only on this
  specific window.
