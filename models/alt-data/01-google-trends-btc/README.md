# 01 — Google Trends Contrarian on BTC

**Hypothesis.** Search interest in "bitcoin" mostly tracks price — when BTC
pumps, more people google it. Treated as a level, the signal carries no
edge. Treated as an *anomaly against its recent average*, extreme values
may flag market sentiment regimes that mean-revert:

- Spikes in attention → retail FOMO → near-term return is *negative*.
- Troughs in attention → disinterest → near-term return is *positive*.

So this is a **contrarian** model on retail attention.

**Data.**

| Source | What | Frequency | Window |
| --- | --- | --- | --- |
| yfinance | BTC-USD daily close, resampled to Sunday-end weekly | weekly | 2021-01-01 to 2024-12-31 |
| Google Trends (via pytrends) | "bitcoin" search interest, global | weekly | 2021-01-01 to 2024-12-31 |

Google Trends data is *relative* (0-100 within the query window), so the
loader queries once over the full window and caches the result. Resolution
is automatic from Google — for a 4-year window we get weekly data.

**Signal.** 4-week rolling z-score of the search-interest series.

- `z < -1.0` → fully long (`position = 1.0`)
- `z > +1.0` → fully short (`position = -1.0`)
- Otherwise → flat (`position = 0.0`)

**Costs.** 2 bp commission + 3 bp slippage per unit of turnover (higher
than equities to reflect crypto-exchange execution).

**Why this model exists.** It's the hello-world for the alt-data pipeline —
proves that a non-price signal can flow through the standard backtest
engine and produce comparable metrics. BTC is the right vehicle because
weekly Google Trends data is reasonably rich for the symbol, and BTC's
volatility means even a modest signal can show measurable impact in the
backtest.

## Reproduce

```bash
uv run python models/alt-data/01-google-trends-btc/backtest.py
```

The first run pulls Google Trends data via pytrends and BTC-USD bars from
yfinance, both cached to `data/processed/`. Subsequent runs read parquet.
Writes [`results/metrics.json`](results/metrics.json) and
[`results/equity_curve.png`](results/equity_curve.png).

## Results — negative

| Metric | Value |
| --- | --- |
| Annualized return | **-20.5%** |
| Sharpe | **-0.30** |
| Max drawdown | **-79.8%** |
| Hit rate | 26.4% |
| Bars (weekly) | 209 |

**The contrarian hypothesis is rejected by the data.** Buying on low
attention and shorting on high attention loses money over 2021-2024. To
sanity-check the conclusion, I also ran the *inverse* (momentum)
direction:

| Variant | Sharpe | Ann. return | Max DD |
| --- | --- | --- | --- |
| Contrarian (this model) | -0.30 | -20.5% | -79.8% |
| Momentum (long/short) | +0.24 | +0.8% | -73.6% |
| Momentum (long-only) | +0.44 | +9.9% | -69.5% |

Even flipping the signal doesn't produce a strategy you'd actually trade
— the long-only momentum variant has a positive Sharpe but still gives up
nearly 70% in drawdown. This is consistent with the interpretation that
search interest is a *coincident* indicator of price rather than a
*leading* one: by the time attention has spiked or cratered, the move
has already happened.

## Why ship a losing model?

Portfolio repos that only show winners are usually overfit. Shipping a
real negative result demonstrates:

1. The backtest engine and alt-data pipeline work end-to-end on a
   non-price signal — this was the *infrastructure* goal of the model.
2. The numbers are honest. Hypotheses that don't pan out get reported.
3. The data is being treated as evidence, not as something to torture
   until it confesses.

## Takeaways

- Search interest is dominated by price (Spearman ρ between the level
  series is very high — see the notebook). Any signal extracted from the
  *level* essentially leaks price information.
- The z-score transformation should have removed that, but with weekly
  resolution and a 4-week lookback the resulting series is still mostly
  momentum-coincident.
- A more promising direction would be: (a) higher-resolution data
  (daily, available with shorter windows or stitched multi-window
  queries), (b) cross-keyword features (e.g. "buy bitcoin" vs "sell
  bitcoin" vs "bitcoin price"), or (c) combining with a real news /
  social signal where the headline can precede price action.

## Caveats

- **Lookahead-safe by construction.** The engine lags the signal one bar;
  z-score at week `t` uses only data through week `t`.
- **Google Trends is relative.** Values are renormalized whenever the
  query window changes. The loader caches the data so this stays stable;
  extending the window later requires recomputing.
- **Small sample.** ~200 weekly bars is too few to put tight confidence
  intervals on the Sharpe — but the drawdown is large enough that the
  conclusion ("contrarian framing fails") is robust to sample noise.
