---
name: Earnings Event-Vol Straddle on SPY
family: options
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars, yfinance_earnings_calendar, dolthub_option_chains]
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
status: negative-result
sharpe_oos: 0.0
max_drawdown: 0.0
---

Long ATM straddle entered into an earnings event on a liquid optionable name. The
straddle is the expression; the **intended** edge is a selection filter — trade only
when the forecast realized move (`expected_move`, the mean of prior earnings-day
absolute returns) exceeds the move implied by the straddle premium
(`implied_move = premium / spot`) by a margin `k > 1`. Phase 1 prices the straddle off a
**synthetic** two-regime vol (an elevated pre-earnings IV that crushes to a lower
post-earnings level, constrained `post_iv < pre_iv`); the realized move comes from real
yfinance daily bars and `expected_move` from **prior earnings events only** (leakage-free,
verified). It is **not tradeable** — the IV it is priced at is assumed, not quoted.
Phase 2 (below) replaces the synthetic surface with **real DoltHub EOD chain quotes**
for the canonical result; the verdict does not change.

> **Verdict up front.** The thorough backtest does **not** demonstrate a tradeable edge.
> The naive (unfiltered) long-straddle program loses significantly; the filtered branch is
> a small, **statistically insignificant** gain whose sign is an **artifact of the assumed
> synthetic IV**; and the selection filter the model is built around is **mechanically
> inoperative in Phase 1**. Viability score is at the bottom.
>
> **Phase 2 (real chains, quote-to-quote) gives that verdict real-data legs.** On market
> quotes the unfiltered program loses **−$412.30/event (p = 0.004)** — the VRP/IV-crush
> headwind measured, not assumed — and the k-gate as designed fired **twice in six
> years**: it compares a one-day forecast against a multi-week implied move, so it is
> **structurally mis-tenored** on real data, and its n = 2 filtered branch can support
> no claim. Status stays `negative-result`.

## Thorough backtest

Reproduce: `uv run python scripts/earnings_straddle_thorough_backtest.py`
(writes `results/thorough_backtest.json`; the canonical `results/validation.json` and
`results/equity_curve.png` are regenerated over the same universe).

- **Universe:** 9 liquid optionable single names — AAPL, AMZN, MSFT, NVDA, TSLA, META,
  GOOGL, NFLX, AMD. (SPY, the nominal `default_ticker`, is an ETF with **no earnings** and
  never trades — a standing quirk of naming the model "on SPY".)
- **Window:** 2020–2026, each name's full cached earnings history — **216 events** (vs. the
  previous 2023–2024 / NVDA-only / 7-trade run).
- **Params:** model defaults `k=1.2, lookback=8, pre_iv=0.45, post_iv=0.25`; 1 bp per-leg
  fee, synthetic `ParametricSpread`; $100k notional, 1 contract/trade.
- **Lens:** trade-level metrics (expectancy, win rate, profit factor, bootstrap on
  per-trade P&L). Per-bar Sharpe is **not** meaningful here — the engine marks equity over
  the full ~1,600-bar window while each straddle is live only ~4 bars, so an annualized
  Sharpe just annualizes a flat curve with one blip. `results/metrics.json` is consequently
  uninformative (it keys off a single event's branch and currently reads all-zeros); the
  real headline is below and in `validation.json`.

### Headline (216 events, baseline params)

| Branch | n | Expectancy | Median | Win | Profit factor | Total P&L | Bootstrap p (H₀: mean = 0) | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Unfiltered** (trade every event) | 216 | −$125.65 | −$215.61 | 32.4% | 0.68 | **−$27,140** | **0.052** | [−$249, +$3] |
| **Filtered** (k-gate fired) | 33 | +$56.69 | **−$117.35** | 39.4% | 1.15 | +$1,871 | **0.778** | [−$324, +$463] |

The filter moves the *relative* numbers in the right direction, but: (1) the filtered edge
is **statistically indistinguishable from zero** (p = 0.78, CI straddles it widely);
(2) the unfiltered branch is the **more** significant of the two (p = 0.052) — the filter
did not produce a significant improvement; and (3) the filtered mean is positive only
because of a **fat right tail** — the *median* filtered trade **loses** $117 and the win
rate is below 50%. The positive expectancy is essentially META + NFLX (21 of the 33 trades).

### Findings

1. **The k-gate is inoperative in Phase 1 — it is a realized-vol screen, not a mispricing
   detector.** `implied_move = straddle_price(spot, K, T≈17/365, pre_iv=0.45)/spot`, and for
   a near-ATM straddle premium/spot depends only on `σ√T`. Because `pre_iv` is a single
   global constant, the implied move is **ticker-independent**: across all 216 events it
   sits at **0.0753–0.0797** (the ~6% spread is pure strike-rounding + tenor jitter,
   carrying *zero* name-specific information). So `expected_move > k·implied_move` collapses
   to the absolute screen `expected_move > ~0.09` — "trade names whose past earnings moved a
   lot." It fires on the high-vol names (META 11, NFLX 10, TSLA 6) and never on the quiet
   ones (MSFT 0, GOOGL 0). **The "filter is the alpha" thesis cannot be tested until a real
   chain supplies name-specific implied vols.**

2. **The result's sign is a free-parameter artifact.** Sweeping the assumed `pre_iv` (fixed
   proportional crush):

   | pre_iv | post_iv | gate fired | Expectancy | Win | PF | Total | p |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 0.30 | 0.167 | 94 | +$472 | 58.5% | **4.60** | +$44,419 | **0.001** |
   | 0.40 | 0.222 | 50 | +$207 | 54.0% | 1.85 | +$10,372 | 0.162 |
   | 0.45 | 0.250 | 33 | +$57 | 39.4% | 1.15 | +$1,871 | 0.778 |
   | 0.55 | 0.306 | 17 | +$6 | 52.9% | 1.02 | +$107 | 0.985 |
   | 0.65 | 0.361 | 9 | −$638 | 44.4% | **0.29** | −$5,745 | 0.182 |

   The strategy swings from "spectacular" (PF 4.60, p = 0.001) to "terrible" (PF 0.29)
   purely as a function of an input no market quoted — a lower assumed IV makes the straddle
   cheaper (lower hurdle) *and* widens the gate (more trades). No edge conclusion survives
   this. (k-margin and lookback sweeps are similarly inconclusive: the best k = 1.5 gives
   PF 2.10 but p = 0.332; k = 2.0 fires on nothing.)

3. **No statistically significant positive edge anywhere; the one significant result is a
   loss.** Benjamini–Hochberg FDR across the names with ≥ 2 trades rejects exactly **one**
   ticker — **NVDA — whose filtered expectancy is −$391.82 with 0/3 winning trades** (its
   gate fired, but the forecast moves didn't materialize). The profitable-looking names
   (META PF 1.70, NFLX PF 1.92) are **not** significant (p = 0.47, 0.39). "Survives FDR"
   here means "distinguishable from zero" — and the only one distinguishable is
   distinguishably *negative*.

4. **Structurally on the wrong side of the volatility risk premium (VRP).** A long ATM
   straddle into earnings pays the VRP (the repo's own `realistic_surface` bakes in
   `vrp = 1.15`, IV > RV) and eats the post-announcement IV crush. The unfiltered
   −$125.65/trade bleed (p = 0.052) is exactly this headwind; the per-name unfiltered losers
   are deeply significant (MSFT t = −5.83, AAPL t = −4.76, GOOGL p = 0.001). The filter's
   job is to overcome the VRP; it has not been shown to.

5. **The unfiltered loss is ~56% synthetic crush, ~44% spread — both are modeling inputs,
   not market facts.** Decomposing the unfiltered book: frictionless (mid fills) it still
   loses **−$69.91/trade** from the modeled IV crush alone; the per-leg double round-trip
   `ParametricSpread` adds **−$55.74/trade**. (This corrects the prior docs' claim that the
   spread is "the dominant cost" — the assumed crush is the larger half, and both are
   synthetic.)

6. **The "calibrated-IV" check does not rescue it — it is circular.** Replacing the fixed
   `pre_iv` with each name's trailing 21-day realized vol makes both branches look great
   (filtered PF 4.74, p = 0.001; *unfiltered* PF 1.69, p = 0.018). But that prices the
   option at realized vol with **zero** volatility risk premium — i.e. it removes the exact
   drag that makes long earnings vol lose, and the filter adds nothing there since the
   *unfiltered* book is already "significant." It is a VRP-stripped lower bound, not evidence
   of edge.

### What is genuinely sound (and why this isn't a 1/10)

- `expected_move` is **leakage-free** (prior events only — verified in `signal.py` /
  `backtest.py`).
- Frictions are modeled honestly: the bid/ask is crossed **per leg**, twice on entry and
  twice on exit (four crossings per straddle round trip).
- The validation harness is real and reusable: centered bootstrap CI, Benjamini–Hochberg
  FDR across the universe, deflated-Sharpe machinery, explicit filtered-vs-unfiltered framing.
- The model labels itself NOT tradeable and reports the insignificant result rather than
  curve-fitting a positive.

### Limitations / not-tradeable blockers

- **Synthetic two-regime `EventVolSurface`** — no real option-chain IV (the decisive
  blocker; see Finding 1).
- **No out-of-sample / walk-forward.** Every number is in-sample; all defaults and all
  sensitivity grids are in-sample, and the deflated-Sharpe `n_trials` hook is **unwired**,
  so the pre_iv × k × calibrated search carries **no multiple-testing penalty**.
- **Universe selection bias:** 9 hand-picked mega-cap tech names over a 2020–2026 AI/tech
  bull run — favorable to large-move outcomes, with no defensives/financials/small-caps/
  delisted names; the pooled bootstrap does not cluster by ticker.
- Expiry approximated as `earnings + 14 calendar days` (no listed-Friday snap); BMO/AMC
  session parsed but unused; BSM delta for American legs.
- Front-matter `sharpe_oos: 0.0` / `max_drawdown: 0.0` are **placeholders**, not measured
  OOS results (per-bar Sharpe is the wrong lens for a sparse event trade).

### Bar for promotion to tradeable

A statistically significant (post-FDR) **positive filtered** edge on **real per-event ATM
chain IV** — `expected_move > k·real_implied_move` beating the VRP **out-of-sample after
real costs** — with listed-expiry snapping and BMO/AMC-correct entry/exit. None of these
exist yet, so this remains a well-built hypothesis, not a strategy. (A short-premium
"mirror" — selling the rich earnings vol the long side overpays for — is the structurally
favored expression and the more natural next experiment.)

## Phase 2 — real chain (DoltHub), quote-to-quote

Reproduce: `uv run python scripts/earnings_straddle_real_chain_backtest.py`
(writes `results/real_chain_backtest.json`; chains cached under
`data/processed/options/dolthub/` — untracked, rebuilt from the free API on a cold run).

Pricing contains **no model**: entry buys the ATM straddle at the real ask, exit sells
both legs at the real bid (DoltHub `post-no-preference/options`, EOD quotes). The
implied move is the real `(call_mid + put_mid) / spot` at entry — **name-specific for
the first time**, so the decisive Phase-1 blocker (assumed IV) is gone. Expiry is a
real listed expiration; strikes require both legs quoted; fees 1 bp per crossing;
spread capped at 20% of mid premium. Same 9 names, same 216 events, same
`k=1.2, lookback=8` defaults as Phase 1. The run is deterministic: a warm re-run
reproduces the JSON byte-for-byte (identical SHA256, ~2.7 s), and cached parquets were
spot-checked against the live API (exact match).

### Headline (113 of 216 events tradeable on real quotes)

| Branch | n | Expectancy | Median | Win | Profit factor | Total P&L | Bootstrap p (H₀: mean = 0) | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Unfiltered** (every tradeable event) | 113 | −$412.30 | −$310.22 | 22.1% | 0.36 | **−$46,590** | **0.004** | [−$683, −$169] |
| **Filtered** (k-gate fired) | 2 | +$3,634.74 | +$3,634.74 | 100% | — (no losers) | +$7,269 | 0.000 — **degenerate** | [+$3,526, +$3,744] |

(Medians computed from the per-event `pnl` values in the JSON's `events` array; at
n = 2 the filtered "median" is just the midpoint of the two trades.)

- **The unfiltered long-straddle program loses significantly on real quotes**: the CI
  sits entirely below zero. Phase 1's −$125.65/event (p = 0.052) was a synthetic
  *understatement* of this headwind.
- **The filtered branch is two trades** — META 2024-02-01 (+$3,743.97) and NFLX
  2024-01-23 (+$3,525.52), both winners. The bootstrap p ≈ 0.000 at n = 2 is
  **degenerate** (resampling two positive numbers can never straddle zero) — **no
  significance claim of any kind is possible from 2 trades.** FDR is empty: no ticker
  reached the 2-filtered-trade minimum.

Skip accounting (113 traded + 103 skipped = 216 ✓):

| Reason | n | Note |
|---|---:|---|
| `contract_missing_at_exit` | 68 | strike row vanished from the re-sampled exit grid; these stay skipped — we refuse to approximate exit quotes off neighbor strikes |
| `no_post_earnings_expiry` | 26 | no expiration visible on both entry and exit chains strictly after the event |
| `no_exit_chain` | 5 | every exit candidate date empty (dataset holes) |
| `no_entry_chain` | 2 | every entry candidate date empty (TSLA 2020 gaps) |
| `spread_over_cap` | 2 | straddle spread > 20% of mid premium |
| `no_quoted_atm`, `strike_grid_mismatch`, `window_out_of_range` | 0 | guards present, never tripped |

### Findings

1. **The structural finding: the k-gate as designed is mis-tenored on real chains — it
   fired twice in six years (2/113 vs Phase 1's 33/216).** The real implied move is the
   **full-tenor** straddle cost: the dataset's holdable expirations put the held tenor at
   16–64 calendar days entry→expiry (median 53; 81 of 113 events at 7–9 weeks), so
   `implied_move` prices *weeks* of movement (pooled mean 0.125), while `expected_move`
   forecasts **one earnings day** (mean 0.065, median 0.060). `expected > 1.2 × implied`
   therefore demands that a single day's forecast move exceed 1.2× a multi-week straddle
   — which happened exactly twice in six years (META 2024-02-01: expected 0.149 vs
   implied 0.084; NFLX 2024-01-23: 0.142 vs 0.116), both monster prints that won.
   Phase 1's ~0.075 synthetic implied accidentally sat low enough to make the gate look
   operative; on real data the *comparison itself* is structurally mismatched. A
   tenor-consistent gate (an event-isolated implied move, or a term-structure
   correction) is the Phase-3 design question.

2. **Implied moves are name-specific now — the decisive Phase-1 blocker is gone.**
   Mean implied move per ticker:

   | AAPL | AMZN | GOOGL | MSFT | NFLX | META | AMD | NVDA | TSLA |
   |---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | 0.0702 | 0.0929 | 0.0976 | 0.0991 | 0.1177 | 0.1262 | 0.1568 | 0.1573 | 0.1763 |

   Quiet names at the bottom, high-vol names at the top — exactly the ordering a real
   surface should produce. Phase 1's ticker-independent 0.0753–0.0797 band is dead.

3. **The unfiltered loss is broad, not one bad name.** 8 of 9 names lose: MSFT
   −$512.86/event (p = 0.000), AAPL −$195.84 (p = 0.000), GOOGL −$709.80 (p = 0.043),
   AMZN −$1,057.55, NFLX −$1,577.01, NVDA −$337.06, TSLA −$325.54, META −$210.14. Only
   AMD is positive (+$181.99, p = 0.50 — noise). This is the volatility risk premium
   plus IV crush, now measured on market quotes instead of bundled into a synthetic
   surface.

4. **The k × lookback sweep is sensitivity, not evidence** — 12 unadjusted looks at the
   same event pool, no multiple-testing penalty. `n_fired` ranges 0–5; the loosest cell
   (k = 1.05, lookback = 8) fires 5 times (PF 2.94, p = 0.277); k ≥ 1.5 fires on at most
   one event. Nothing is remotely significant, and with so few firings nothing could be.

### Dataset constraints (DoltHub `post-no-preference/options`) and mitigations

All discovered empirically during the run; none are hidden in the results:

- **EOD quotes**, with **Mon/Wed/Fri-only coverage before ~Oct 2024** (daily after).
  Mitigation: entry/exit snap to the nearest covered chain date within a bounded window
  (entry alternates nearer/farther, never closer than 1 bar before the earnings bar;
  exit only moves forward). Offsets actually used in this run:
  `entry_lead_used {3: 95, 2: 17, 4: 1}`, `exit_offset_used {1: 82, 2: 30, 3: 1}` —
  recorded per event.
- **Each (date, symbol) lists only ~3 Friday expirations** at tenor-anchored slots
  (~2w/4w/7w out); the front week is **never** listed, and the visibility window rolls
  day to day. Mitigation: expiry is chosen from the **intersection** of entry-chain and
  exit-chain expirations — a workaround for *marking* the position at both dates, **not
  price lookahead** (the contract traded continuously; the selection conditions only on
  which rows the dataset publishes, never on quote values). Stated side effect: the held
  tenor is multi-week (median 53 days) rather than the nominal 2-week spec.
- **The ~27-strike band is re-sampled daily around spot** and the grid phase can shift,
  so a strike present at entry can be absent at exit. Mitigation: none possible without
  fabricating quotes — those 68 events are skipped as `contract_missing_at_exit`.
- **Strikes are contemporaneous, never retro-adjusted for splits**, while yfinance
  closes are fully adjusted. Mitigation: an explicit `SPLITS` table in the runner
  rescales spot to contemporaneous dollars (verified against every cached chain), plus
  a 15% strike-grid guard that skips any event where the factor could be wrong (it
  never tripped).
- **Pre-rename META is keyed under FB.** Mitigation: the loader aliases dates before
  2022-06-09.
- **Known holes**: 2024-08-01..06 are empty table-wide; TSLA has 2020 gaps. Mitigation:
  the 7-reason skip taxonomy, with per-ticker counts in the JSON.

### What would promote it further

- **A tenor-consistent gate on the same data** (Phase 3): isolate the event component
  of the implied move, or correct the multi-week straddle cost for the term structure,
  so `expected_move` and `implied_move` price the same horizon — then re-test
  filtered-vs-unfiltered.
- **Out-of-sample via the accruing yfinance snapshots**
  (`scripts/collect_chain_snapshots.py` runs daily; the snapshots are the only true
  point-in-time OOS data this model will ever have).
- **The short-premium mirror** (unchanged from Phase 1, now with a measured number):
  the −$412/event unfiltered bleed is what the short side would have *collected* —
  before its unmodeled tail and margin risk.

## Viability: 3 / 10

Scored as a **tradeable strategy**, not as infrastructure.

- **Against (−):** structurally short the volatility risk premium (the losing side); the
  selection "alpha" is mechanically inoperative in Phase 1 (ticker-independent implied move
  → a vol screen); the filtered edge is statistically insignificant (p = 0.78) and
  tail-driven (negative median); the only FDR-significant name is a *loser*; the headline
  sign is an artifact of an uncalibrated assumption (PF 4.60 → 0.29 across the pre_iv sweep);
  not tradeable as built; no OOS.
- **For (+):** sound, honestly reported economics; leakage-free signal; genuinely good,
  reusable validation scaffolding; and a concrete real-chain path that *could* turn the same
  code into a real test.

Phase 2 does not move the score. Against: the unfiltered loss became **real and
significant** (−$412.30/event, p = 0.004 on market quotes), and the filter the model is
built around proved **structurally mis-tenored** on real chains (2 fires in six years —
no measurable selection edge). For: the decisive Phase-1 blocker (assumed IV) is gone,
the validation now runs on real quotes end-to-end, and the failure mode is a precise,
fixable design question (a tenor-consistent gate) rather than an artifact. The
infrastructure improved; across both phases the **strategy's demonstrated, tradeable
edge is still essentially zero**, which is what this 3/10 reflects.
