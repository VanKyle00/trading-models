# Earnings Event-Vol (Long Straddle) Options Model — Design Spec

**Date:** 2026-06-08
**Status:** Approved (pending written-spec review)
**Scope:** A new options model, `03-earnings-straddle`, plus a small earnings-
calendar loader and a validation-harness upgrade. Inspired by Poudel (2025),
*Small-Cap Stock Trading Strategies for Retail Traders* (SSRN 5921742),
Strategy Family E. Real historical options-chain ingestion remains **SP3** and
is out of scope here; this model runs on a synthetic straddle first and a free
forward-snapshot dataset second (see Data Plan).

## Motivation

The user asked to turn SSRN paper 5921742 into an options trading model. The
paper is a small-cap **equity** dissertation, not an options paper, so the task
is translation, not transcription. Its Strategy Family E *describes* a
long-strangle-around-earnings volatility trade and then explicitly declines to
implement it: *"For simplicity, this section focuses on equity-based event
trading rather than options."* That punted options strategy is the gap this
model fills.

Three things from the paper carry over; the rest does not:

1. **Strategy E (event vol)** is the seed — taken to its intended options form.
2. **Chapter 4 validation methodology** (walk-forward, bootstrap t-test,
   Benjamini-Hochberg FDR, the "why backtests fail" checklist) is the most
   transferable contribution and upgrades the existing SP1 harness.
3. **Chapter 6 risk framework** (vol-parity / fixed-fractional sizing,
   no-trade filters, kill-switch) informs sizing and gating.

Deliberately **not** carried over: the small-cap universe (small-cap *options*
are illiquid — wide spreads, sparse strikes — so this model trades only liquid
optionable names); and the intraday strategies (ORB, gap-and-go), which fit
neither the options framing nor a daily-bar workflow.

## The core insight (defines the model)

The paper's Strategy E was its **weakest** result (0.54 OOS Sharpe, lowest of
the six). That is the tell, not a flaw to ignore: naively buying a straddle into
*every* earnings loses, because the implied-vol premium baked into the straddle
usually exceeds the realized post-earnings move, and the position bleeds through
the post-announcement IV crush.

So the model's edge is **not** "own vol into earnings." It is a **selection
filter**: enter a straddle **only when the forecast realized move exceeds the
implied move priced into the straddle by a margin.** The filter is the alpha;
the straddle is the expression. A model that cannot beat the premium on average
should report that honestly rather than curve-fit around it.

## Decisions (locked during brainstorming)

- **Direction: A — earnings event-vol (long straddle).** Not the regime-
  directional (B) or methodology-only (C) alternatives.
- **Short-hold event trade, not a 2w–6mo swing.** Option *tenor* is the nearest
  listed expiry strictly after the earnings date; *holding period* is ~T−3
  trading days to T+1. The earlier "2w–6mo" framing was for options models in
  general; this event model is intentionally short-hold.
- **Liquid optionable underlyings only** (e.g. large caps / liquid single names
  with weekly options). Universe is a configurable watchlist, not the paper's
  small-cap screen.
- **Selection filter is mandatory**, not optional polish. `expected_move >
  implied_move × k`, `k > 1`, `k` a model parameter.
- **Data is phased, free-first** (matches the SP2 synthetic-first precedent).
  Phase 1 synthetic, Phase 2 free forward-snapshot, Phase 3 optional paid. No
  paid data is required to build, test, and validate the pipeline.
- **Validation methodology lands in the existing harness**, not a parallel one.
  Bootstrap t-test and Benjamini-Hochberg FDR are added to `tradinglib/
  validation/`; the model is judged through walk-forward across earnings seasons.

## Component 1 — Earnings calendar loader (`tradinglib/loaders/events/earnings.py`)

A small, pluggable provider that returns point-in-time scheduled earnings dates
(and BMO/AMC session timing where available) for a list of tickers.

- **Default source:** yfinance (`Ticker.earnings_dates` / `.get_earnings_dates`),
  free, no key. Forward-looking schedule plus the limited history yfinance
  exposes.
- **Contract:** `get_earnings_dates(tickers, start, end) -> DataFrame` with
  columns `[ticker, earnings_datetime, session]` where `session ∈
  {bmo, amc, unknown}`. Canonicalized and written to
  `data/processed/events/earnings/...` parquet, mirroring the yfinance equities
  loader pattern documented in `docs/data-sources.md`.
- **Point-in-time discipline (Ch. 4):** never use an earnings date or session
  label that would not have been knowable at decision time. The loader records
  the schedule as-of snapshot dates so Phase 2 forward data is leak-free by
  construction.
- **Pluggable:** the provider is an interface so a paid calendar (FMP, Nasdaq,
  EDGAR 8-K derivation) can replace yfinance later without touching the model.

## Component 2 — Straddle construction (reuse `tradinglib/options/`)

No new pricing code. The model assembles an **ATM straddle** (long call + long
put, same strike ≈ spot, same expiry) using existing `instruments.py` and
`pricing.py`:

- Strike = nearest listed strike to spot at entry (ATM).
- Expiry = nearest listed expiry **strictly after** the earnings datetime.
- Greeks (vega, theta) tracked via `bs_greeks` for diagnostics and the no-trade
  filters. Rates and dividends are passed through (non-negligible only for the
  rare longer-tenor case; usually tiny at this horizon).

## Component 3 — The selection signal (`models/options/03-earnings-straddle/`)

The alpha. Computed per candidate earnings event:

- **Implied move:** `straddle_price / spot` (the standard ATM-straddle implied-
  move approximation), equivalently `≈ atm_iv × sqrt(T)`. This is what the
  market charges.
- **Expected move:** a forecast of the realized earnings-day absolute move.
  Baseline forecast = a robust statistic (e.g. median) of the underlying's own
  past N earnings-day abs returns. Optionally blended with an IV term-structure
  signal later; baseline ships first (YAGNI).
- **Entry rule:** go long the straddle iff `expected_move > implied_move × k`
  (`k` parameter, `k > 1`). Otherwise skip the event.
- **Parameters** (small grid, ranges chosen for FDR-honest counting):
  entry lead days (T−k_entry), exit offset (T+k_exit), `k` (edge margin),
  N (earnings lookback for the forecast), min-liquidity gate.

## Component 4 — Entry/exit & accounting (reuse `backtest/options_engine.py`)

- **Enter** at the configured lead (default T−3 trading days), priced at entry-
  day close (next-open fills per the SP1 default where an opens series exists).
- **Exit** at the configured offset (default T+1), mark-to-market through the
  options engine's expiry/roll/MTM machinery.
- **Frictions (reuse SP2):** apply the synthetic bid/ask spread model — and note
  a straddle pays it **twice** (two legs), entry and exit. This is the dominant
  cost and must not be understated; it is a first-class line in results.

## Component 5 — Risk framework (Ch. 6, scoped down)

- **Sizing:** long options ⇒ defined risk (max loss = premium paid). Size by
  fixed fraction of capital per trade against the premium; vol-parity scaling is
  available as a parameter but fixed-fractional is the default (YAGNI).
- **Portfolio cap:** maximum number of concurrent open earnings straddles.
- **No-trade filters:** skip events with missing/!valid IV, illiquid chains
  (no near-ATM strike, no post-earnings expiry), or spread wider than a cap.

## Component 6 — Validation upgrade (`tradinglib/validation/`, from Ch. 4)

Extends the existing SP1 harness rather than forking it:

- **Benjamini-Hochberg FDR control** over the cross-ticker hypothesis set —
  testing many names is exactly the multiple-testing problem the paper warns
  about; report which tickers survive FDR at α.
- **Bootstrap t-test** (non-parametric) for the strategy's per-trade returns,
  which are fat-tailed and few — Student's t is inappropriate.
- **Walk-forward across earnings seasons:** parameters chosen on in-sample
  seasons, frozen on out-of-sample seasons; the parameter grid feeds the
  existing Deflated Sharpe trial count.
- **Trade-level metrics** (Ch. 4.3.5): win rate, profit factor, expectancy,
  average win/loss, average hold — alongside the existing Sharpe/Deflated Sharpe/
  max-drawdown set.

## Data Plan (phased, free-first)

1. **Phase 1 — synthetic, now, zero cost.** Build and unit-test the *entire*
   pipeline against a **synthetic straddle**: the BS pricer driven by an explicit
   pre-earnings IV (elevated) and a parameterized post-earnings IV crush, with
   the realized move taken from actual yfinance underlying bars. This validates
   the signal logic, sizing, frictions, and the new FDR/bootstrap stats today.
   Clearly labeled **not-yet-tradeable**, exactly as the repo already treats
   synthetic vol.
2. **Phase 2 — free, forward-accumulating.** A snapshot collector (yfinance
   `.option_chain()` + the earnings calendar) records real ATM straddle prices
   and IV for a liquid-earnings watchlist going forward. Real backtest after a
   couple of earnings seasons accrue. No cost; history grows over time.
3. **Phase 3 — optional, paid.** Polygon / ORATS EOD options history for
   immediate deep backtests, if the user wants real results before Phase 2
   accrues. This is the SP3 chain-loader, deferred.

## Out of scope

- SP3 historical options-chain ingestion (Phases 2–3 above are deliberately
  light-touch; the full chain loader is its own design cycle).
- The paper's other five strategy families and its intraday machinery.
- Strangles, calendars, and other structures (straddle only; YAGNI).
- IV-term-structure blending in the forecast (baseline historical-move forecast
  ships first).
- README / MODELS.md regeneration (per-model artifacts hold the numbers this
  cycle; README's Current-models table is updated when the model is shipped).

## Success criteria

- The synthetic Phase-1 pipeline runs end-to-end: calendar → candidate events →
  selection filter → straddle entry/exit → frictioned P&L → validation report.
- Unit tests cover: implied-move and expected-move computation, the `k`-margin
  entry gate, double-spread friction on the two legs, no-trade filters, and the
  new FDR / bootstrap functions.
- The validation report distinguishes filtered vs. unfiltered straddle P&L,
  demonstrating that the selection filter is what (if anything) creates the edge
  — and reports honestly when it does not beat the premium.
