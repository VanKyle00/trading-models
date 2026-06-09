# Earnings Event-Vol Straddle on SPY

Long ATM straddle entered into an earnings event on a liquid optionable name.
The straddle is just the expression; **the edge is the selection filter**, not
"owning vol into earnings." We trade only when the forecast realized move
(`expected_move`, the mean of prior earnings-day absolute returns) exceeds the
move implied by the straddle premium (`implied_move = premium / spot`) by a
margin `k > 1`. The model compares the **filtered** branch (trade only when the
gate fires) against the **unfiltered** branch (trade every event).

> **Phase-1 caveat (see [`model.md`](model.md)).** In Phase 1 the gate **cannot**
> demonstrate selection alpha: because the synthetic `pre_iv` is a single global
> constant, the implied move is ticker-independent (~0.075–0.080 for every name),
> so `expected_move > k·implied_move` degenerates into a pure realized-vol screen.
> The thorough backtest (216 events) finds the filtered edge **statistically
> insignificant** (p = 0.78) and its sign an **artifact of the assumed IV**. Treat
> "the filter is the alpha" as a hypothesis for the real-chain phase, not a result.

## What it tests

- The k-margin selection gate (`expected_move > implied_move * k`) as the source
  of edge, evaluated filtered-vs-unfiltered.
- Double-leg spread frictions through the real options engine: a straddle is two
  legs, and the bid/ask spread is charged **per leg**, so it is paid **twice on
  entry and twice on exit**. This four-legged round-trip spread is a major cost
  (~44% of the unfiltered loss in the thorough run), but the modeled IV crush is
  the larger half (~56%) — both are synthetic inputs, not market facts.
- Cross-event significance: a non-parametric bootstrap CI on pooled per-trade
  P&L plus Benjamini-Hochberg FDR control across the watchlist (so a single
  lucky ticker does not masquerade as edge).

## Results (thorough backtest — 216 events, 9 names, 2020–2026)

Universe: AAPL, AMZN, MSFT, NVDA, TSLA, META, GOOGL, NFLX, AMD (SPY has no
earnings). Full per-ticker breakdown and sensitivity tables in
[`model.md`](model.md) and `results/thorough_backtest.json`.

| Branch | n | Expectancy | Median | Win | PF | Total | Bootstrap p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unfiltered (every event) | 216 | −$125.65 | −$215.61 | 32.4% | 0.68 | **−$27,140** | 0.052 |
| Filtered (k-gate) | 33 | +$56.69 | **−$117.35** | 39.4% | 1.15 | +$1,871 | **0.778** |

**Findings:**

- The filter lifts the *relative* numbers, but the filtered edge is
  **statistically insignificant** (p = 0.78, CI crosses zero) and the unfiltered
  branch is actually the *more* significant of the two (p = 0.052). The positive
  filtered mean is a **fat tail** — the median trade loses $117 and two names
  (META, NFLX) supply the entire positive contribution.
- **The k-gate is a realized-vol screen, not a mispricing detector** in Phase 1:
  the synthetic implied move is ticker-independent (~0.075 for every name), so
  the gate just selects names whose past earnings moves were large.
- **The result's sign is an artifact of the assumed `pre_iv`** — profit factor
  swings 4.60 (p = 0.001) → 0.29 as `pre_iv` goes 0.30 → 0.65.
- **No significant positive edge anywhere.** The only FDR survivor (NVDA) is a
  significant *loser* (−$391.82, 0/3 wins).
- Structurally **short the volatility risk premium** — the unfiltered
  −$125.65/trade bleed is that headwind (≈56% synthetic crush, ≈44% spread).

## Viability: 3 / 10

Scored as a *tradeable strategy*, not as infrastructure. It is short the VRP, the
selection "alpha" is mechanically un-demonstrable in Phase 1, the edge is
statistically insignificant and artifact-driven, and it is not tradeable as built
(synthetic vol, no real chain). It avoids a 1–2 only because the economics are
sound and honestly reported and the validation scaffolding (leakage-free signal,
bootstrap, FDR, deflated-Sharpe machinery) is genuinely good, with a concrete
real-chain path forward. Full rationale at the bottom of [`model.md`](model.md).

## Phase 1 is synthetic — NOT yet tradeable

Pricing uses an explicit pre-earnings IV and a parameterized post-earnings crush
(`EventVolSurface`, constrained `post_iv < pre_iv` so the synthetic premium can
never be tuned into a fake IV expansion). The **realized** move comes from real
yfinance daily bars; expected move is computed from **prior earnings events
only** (no leakage). This mirrors the repo's SP2 synthetic-frictions treatment.
Real forward chain snapshots (Phase 2, free) and paid chain history (Phase 3)
are out of scope here.

## Deferred (stated, not silently dropped)

- **Greeks diagnostics** (vega/theta) from Component 2: the no-trade filters gate
  on implied-move validity, a post-earnings expiry, and the spread cap only —
  they do not require greeks this phase.
- **Walk-forward across earnings seasons**: the existing `validation/walk_forward`
  harness is built on the vectorized equity engine and is incompatible with the
  `OptionsEngine` path this model uses. An options-aware walk-forward is a
  separate design cycle; the Deflated-Sharpe `n_trials` hook is the future wiring
  point for a parameter grid.
- **Listed-expiry snap**: expiry is approximated as `earnings + 14 calendar days`
  rather than snapped to the nearest listed weekly Friday; the snap arrives with
  the real chain.

## Reproduce

```bash
uv run python models/options/03-earnings-straddle-spy/backtest.py
```

Writes `results/metrics.json`, `results/validation.json`, and
`results/equity_curve.png`. The run is data-optional: tickers whose bars or
earnings dates are unavailable are skipped.

> `metrics.json` keys off a single event's branch and is **uninformative** as a
> headline (it currently reads all-zeros). The meaningful trade-level result lives
> in `validation.json`; the full event study + sensitivity sweeps are in
> `results/thorough_backtest.json` (reproduce with
> `uv run python scripts/earnings_straddle_thorough_backtest.py`).

## Reading `results/validation.json`

- `per_ticker[<ticker>]` — the filtered vs unfiltered branch for each ticker
  (final equity, trade P&L, the implied/expected moves, and the engine metrics).
- `pooled_filtered` — the bootstrap test over **all** filtered per-event P&Ls:
  `bootstrap_t_stat`, the `[bootstrap_ci_lower, bootstrap_ci_upper]` CI, and the
  centered `bootstrap_p_value`.
- `fdr` — per-ticker bootstrap p-values with Benjamini-Hochberg `rejected` flags
  and the BH `threshold`. **`rejected == true` only means the mean is
  distinguishable from zero — the bootstrap p-value is two-sided, so a rejection
  can be a significant _loss_.** In the thorough run the **only** survivor is
  NVDA, whose filtered expectancy is **−$391.82 (0/3 wins)** — a significant
  loser, not a confirmed edge. No name shows a significant _positive_ edge. A
  ticker with fewer than 2 events gets the sentinel p-value 1.0 and cannot be
  rejected.
- `trade_metrics` — pooled trade-level stats (`n_trades`, `win_rate`,
  `profit_factor`, `expectancy`, `avg_win`, `avg_loss`).
