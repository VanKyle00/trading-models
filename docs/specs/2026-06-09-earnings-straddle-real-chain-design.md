# Earnings Straddle Phase 2 — Real Option-Chain Data — Design Spec

**Date:** 2026-06-09
**Status:** Approved (pending written-spec review)
**Scope:** Wire real option-chain data into `models/options/03-earnings-straddle-spy`,
replacing the synthetic `EventVolSurface` path for the canonical results: a DoltHub
historical chain loader, a yfinance forward-snapshot collector, and a quote-to-quote
real-chain event backtest run through the existing validation harness. Follows the
Phase-2 step of the original design
(`docs/specs/2026-06-08-earnings-straddle-options-model-design.md`).

## Motivation

The Phase-1 thorough backtest (216 events, 9 names) established that the model's core
thesis — "the selection filter is the alpha" — **cannot be tested on synthetic vol**:

1. The implied move is computed from a single global `pre_iv`, so it is
   ticker-independent (~0.075 for every name) and the k-gate degenerates into a
   realized-vol screen.
2. The headline sign is an artifact of the assumed IV (profit factor 4.60 → 0.29
   across the `pre_iv` sweep).
3. Both major costs (IV crush, spread) are modeling inputs, not market facts.

`model.md` states the bar for promotion: a post-FDR significant positive filtered edge
on **real per-event ATM chain IV**, out-of-sample, after real costs. This cycle supplies
the real-chain data and the real-cost backtest; OOS accrual starts with the snapshot
collector.

## Decisions (locked during brainstorming, 2026-06-09)

- **Scope: core + snapshot collector.** The short-premium mirror, OOS season holdout,
  deflated-Sharpe `n_trials` wiring, and GUI changes are explicitly deferred.
- **P&L path: quote-to-quote.** Entry buys the ATM call and put at the real ask; exit
  sells both at the real bid. No BSM repricing, no vol surface, no `ParametricSpread`
  anywhere in the real-chain path — frictions are real by construction. The
  `ChainSurface`-through-`OptionsEngine` alternative was rejected: it reintroduces model
  error (BSM on American legs, IV interpolation) exactly where direct quotes exist.
- **Historical source: DoltHub `post-no-preference/options`** via the free SQL API.
  Verified empirically 2026-06-09: schema
  `option_chain(date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta,
  gamma, theta, vega, rho)` with PK `(date, act_symbol, expiration, strike, call_put)`;
  live data through 2026-06-08; coverage back to at least 2020.
- **Query discipline:** every API query filters on exact `date` **and** `act_symbol`
  (the PK prefix). Anything else scans and times out (`context deadline exceeded`,
  observed). No `DISTINCT`/aggregate scans against the live API.
- **Expiry constraint (verified):** the dataset carries only 3–4 expirations per
  (date, symbol) — select weeklies + monthlies roughly 2–7 weeks out. The straddle uses
  the nearest *available* expiration strictly after the earnings datetime, which is
  real and listed but may be 1–5 weeks post-earnings rather than the front weekly. This
  is documented in results, not hidden.
- **Forward source: yfinance `option_chain()`** (verified live 2026-06-09: per-strike
  bid/ask, IV, volume, OI). Snapshots share the historical loader's canonical schema so
  the backtest can consume either source later.

## Component 1 — DoltHub chain loader (`tradinglib/loaders/options/dolthub.py`)

- **Contract:** `load_chain(ticker: str, date: str | datetime, *, refresh: bool = False)
  -> DataFrame` — one (ticker, trading date) → that day's full available chain.
- **Canonical schema:** `[date, ticker, expiration, strike, right, bid, ask, iv]` with
  `right ∈ {"call", "put"}`, dates as tz-naive datestamps, numerics as float (DoltHub
  returns decimals as strings; the loader coerces). Greeks are dropped this cycle
  (YAGNI — the no-trade filters do not need them).
- **API:** `GET https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master`
  with `q=SELECT ... FROM option_chain WHERE date='<d>' AND act_symbol='<t>'`. Timeout
  60 s, one retry on transient failure, a short politeness sleep between live calls.
- **Cache:** `data/processed/options/dolthub/<ticker>/<date>.parquet`, mirroring the
  earnings-loader pattern. A cached file is never re-fetched unless `refresh=True`.
  An empty API result writes an empty canonical frame (so the miss is cached too) and
  the consuming event is skipped with a reason.
- **Errors:** HTTP/parse errors raise after the retry; the loader never fabricates rows.
- **Tests (network mocked, repo convention):** canonicalization from a captured API
  response fixture, string→float coercion, cache round-trip, empty-result handling.

## Component 2 — Forward snapshot collector (`tradinglib/loaders/options/yf_chain.py` + `scripts/collect_chain_snapshots.py`)

- **Contract:** `snapshot_chains(tickers: list[str]) -> DataFrame` — fetches every
  listed expiration ≤ 45 calendar days out per ticker via yfinance `option_chain()`,
  canonicalizes to the Component-1 schema **plus a `spot` column** (from `fast_info`),
  and writes `data/processed/options/yf_snapshots/<ticker>/<snapshot-date>.parquet`.
- **Idempotent:** if today's snapshot file exists, the ticker is skipped (no refresh
  flag needed — a snapshot is point-in-time by definition).
- **Script:** `scripts/collect_chain_snapshots.py` runs the collector over the model's
  9-name watchlist (AAPL, AMZN, MSFT, NVDA, TSLA, META, GOOGL, NFLX, AMD) and prints a
  one-line summary per ticker. Intended to be scheduled daily (Windows Task Scheduler /
  cron); scheduling itself is documented in `docs/data-sources.md`, not built.
- **Tests (yfinance mocked):** canonicalization, spot column, 45-day expiry cutoff,
  idempotency.

## Component 3 — Real-chain event backtest (`models/options/03-earnings-straddle-spy/real_chain.py` + `scripts/earnings_straddle_real_chain_backtest.py`)

Core event mechanics live in `real_chain.py` (importable for tests, like `signal.py` /
`strategy.py`); the script is a thin runner.

**Per earnings event** (same bar-counting as `strategy.py`: earnings bar = first bar at
or after the earnings datetime; entry = `entry_lead`(3) bars before it; exit =
`exit_offset`(1) bars after it):

1. **Entry chain:** `load_chain(ticker, entry_date)`. Expiry = nearest available
   expiration strictly after the earnings datetime. Strike = nearest to the entry-date
   close (spot, from the existing yfinance daily bars) with **both** legs quoted
   (`bid > 0` and `ask > 0` on call and put).
2. **Entry economics:** `cost = call_ask + put_ask`; per-leg fee 1 bp (consistent with
   Phase 1); `implied_move = (call_mid + put_mid) / spot` — real and name-specific, via
   the unchanged `signal.implied_move`.
3. **Gate:** `signal.expected_move` (prior events only, unchanged) and
   `signal.passes_filter(expected, implied, k)` with default `k=1.2`, `lookback=8`.
   Chain-tradeability via `signal.tradeable_event` with the straddle's relative
   spread (`(asks − bids) / mid premium`) capped at `max_spread_frac=0.20` — the
   value the existing signal tests use; Phase 1 never wired this cap into a backtest,
   so this is its first live use. The cap is a parameter recorded in run metadata.
4. **Exit chain:** `load_chain(ticker, exit_date)`; the *same* contracts (expiry,
   strike) are valued at `call_bid + put_bid`. A zero exit bid is a **valid
   total-loss outcome** (post-crush worthless leg), not missing data — only absent
   contract rows at exit count as missing. P&L =
   `(exit_value − entry_cost) × 100 × contracts − fees`, 1 contract per trade,
   matching Phase 1.
5. **Skip reasons are first-class.** An event that cannot trade is skipped with exactly
   one counted reason: `no_entry_chain` (no rows for the entry date),
   `no_post_earnings_expiry`, `no_quoted_atm` (no strike with both legs quoted at
   entry), `spread_over_cap`, `no_exit_chain` (contract rows absent at exit). The
   report prints all counts — no silent truncation.

**Branches:** unfiltered (every tradeable event) and filtered (k-gate fired), exactly
the Phase-1 framing.

**Aggregation:** reuses the existing harness untouched — `bootstrap_t_test`,
`benjamini_hochberg_fdr`, `trade_metrics` — pooled and per-ticker, plus a k × lookback
sensitivity sweep. There is no `pre_iv` to sweep: the Phase-1 free-parameter artifact
is structurally gone.

**Output:** `models/options/03-earnings-straddle-spy/results/real_chain_backtest.json`
with: headline filtered-vs-unfiltered table, per-ticker breakdown, skip-reason counts,
sensitivity sweep, and run metadata (universe, window, params, data source).

**Universe/window:** the same 9 names; every earnings event (from the existing earnings
loader caches) whose entry and exit dates have DoltHub coverage. First live run is
~2 × 216 ≈ 432 API calls; everything after is parquet-cached.

**Tests:** expiry selection (nearest strictly-after; none available), ATM strike
selection (both-legs-quoted requirement), quote-to-quote P&L arithmetic including fees,
each skip reason, and gate integration with mocked chains.

## Component 4 — Documentation

- `model.md` / `README.md`: a "Phase 2 — real chain" results section with the
  real-quote headline table, skip-reason accounting, and a re-verdict (viability
  re-scored against the real numbers, whatever they say). Front-matter updated if the
  status changes.
- `docs/data-sources.md`: entries for the DoltHub options database (API, PK query
  discipline, coverage, expiration subset caveat) and the yfinance snapshot collector
  (schema, cadence, scheduling note).

## Error-handling philosophy

Data-optional throughout, like the rest of the repo: missing or unusable data skips the
event with a logged, counted reason. The backtest never substitutes synthetic values
into the real-chain path.

## Out of scope (deferred, stated)

- Short-premium mirror (needs margin/assignment modeling the engine lacks).
- OOS season holdout and deflated-Sharpe `n_trials` wiring (next cycle, once real-chain
  in-sample results exist).
- Ticker-clustered bootstrap.
- GUI / `run_for_gui` changes — the Streamlit path keeps the synthetic surface and its
  existing not-tradeable labeling.
- Consuming yfinance snapshots in the backtest (the collector only accumulates data this
  cycle; a source switch lands when enough history exists).

## Success criteria

- `uv run python scripts/earnings_straddle_real_chain_backtest.py` runs end-to-end on
  cached data and writes `results/real_chain_backtest.json` with both branches, FDR,
  skip counts, and the k × lookback sweep.
- The k-gate consumes a name-specific real implied move (visibly different across
  tickers, unlike Phase 1's ~0.075 constant).
- `uv run python scripts/collect_chain_snapshots.py` writes today's canonical snapshot
  parquets for the watchlist and is idempotent on re-run.
- All new unit tests pass with the network fully mocked; `uv run pytest` stays green.
- `model.md` / `README.md` report the real-chain results honestly, including a negative
  result if that is what the data says.
