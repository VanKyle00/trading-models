# Options Realistic Frictions — Synthetic Vol Surface & Spread — Design

**Date:** 2026-06-07
**Status:** Approved (design); pending spec review
**Approach:** Add a synthetic, calibrated **vol surface** and a **bid/ask spread
model** to the existing options backtest engine so that 2-week–6-month option
strategies can be tested under realistic frictions — without a paid options-chain
data subscription. No new market data; the surface is anchored to the
underlying's own realized volatility.

## Goal

Make `run_options_backtest` produce *trustworthy* P&L for swing/positional
option trades (2w–6mo holds: directional ATM/OTM, vol/hedged, and vertical /
calendar spreads). Today the engine prices every leg off a **single constant
`vol`** and fills at the Black-Scholes **mid** with only a flat-bps cost. For
multi-week horizons that is fantasy: it ignores volatility regimes, skew, term
structure, and the bid/ask spread — which is most of the real-world drag on
options.

This is **SP2** (realistic frictions) of the validation effort. It also makes the
originally-scoped **SP3** (a paid historical options-chain loader) unnecessary
*for now*: the surface/spread interfaces are modular, so a real-chain source can
replace the synthetic one later as a drop-in.

### Honest scope of the claim

This is a **stress / plausibility model, not a market-calibrated one.** It tells
you whether an edge survives *realistic-shaped* vol regimes and frictions; it
**cannot** reproduce the exact historical P&L of a specific contract. That
requires real chain data (deferred SP3). The modular interfaces make that a
later drop-in, not a rewrite.

## Background — what exists today

`tradinglib/backtest/options_engine.py` (`OptionsEngine` / `run_options_backtest`):

- Constructed with a single scalar `vol: float`.
- `_price_leg` / `_leg_delta` price every leg via `bs_price` / `crr_price` /
  `bs_greeks` using `self.vol`.
- `add_leg`, `close_all_options`, `hedge_to_delta` transact at the **model mid**
  and apply a flat `cost_rate = (fee_bps + slippage_bps) / 10_000` on notional.

`tradinglib/options/` provides the untouched pricing/instrument primitives this
design builds on:

- `bs_price(right, spot, strike, t_yrs, vol, rate)`,
  `crr_price(..., style=)`, `bs_greeks(...) -> Greeks`, `implied_vol(...)`.
- `OptionLeg(right, strike, expiry, quantity, style)`, `CONTRACT_MULTIPLIER = 100.0`.

The seed model `models/options/01-delta-hedged-long-option-spy` runs at a
constant `IMPLIED_VOL = 0.18`.

## Design

Two new modules plus a surgical change to the engine. The pricing math
(`bs_price`/`crr_price`/`bs_greeks`) is **not** touched — only the *vol input*
and the *fill price* change.

```
tradinglib/options/
  surface.py     # NEW: VolSurface protocol, FlatSurface, ParametricSurface, realized-vol calibration
  spread.py      # NEW: SpreadModel protocol, NoSpread, ParametricSpread
  pricing.py     # unchanged
  instruments.py # unchanged
  simulate.py    # unchanged (surface-awareness is out of scope for v1)
tradinglib/backtest/options_engine.py   # CHANGED: price & fill via surface + spread
models/options/02-directional-call-spy/ # NEW: demo proving realism on a directional 2w–6mo trade
```

### 1. Vol surface — `tradinglib/options/surface.py`

```python
class VolSurface(Protocol):
    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float: ...
```

**`FlatSurface(vol)`** — returns the constant `vol` for every query. Reproduces
today's behavior exactly; used for backward-compat and the constant-vol
educational seed model.

**`ParametricSurface`** — implied vol as a separable product:

```
IV(strike, expiry, t) = atm_vol(t) · term_factor(dte) · skew_factor(m, dte)
```

where `dte = (expiry - t).days` and `m = log(strike / spot)` (log-moneyness).

- **`atm_vol(t)` — time-varying, calibrated to the underlying.** Trailing
  realized volatility of close-to-close log returns over a fixed window
  (default 21 bars, annualized by `√252`), multiplied by a
  volatility-risk-premium constant `vrp` (default `1.15`, since implied
  historically runs above realized). Precomputed once as a `pd.Series` indexed
  like the price series; the surface looks it up with `.asof(t)`. **This is what
  replaces the constant-0.18 fantasy** — ATM IV now rises in stressed regimes
  and falls in calm ones.
- **`skew_factor(m, dte)`** — quadratic smile in log-moneyness:
  `1 + b·m + c·m²`, with `b < 0` (equity skew: OTM puts richer, OTM calls
  cheaper) and `c ≥ 0` (smile curvature). The skew slope `b` flattens with
  longer `dte` (`b_eff = b / (1 + k·dte/365)`), matching the empirical fact that
  long-dated skew is shallower.
- **`term_factor(dte)`** — mild slope, normalized to `1.0` at the realized-vol
  window length so short-dated ATM ≈ realized vol. Default gently upward-sloping
  (short-dated slightly cheaper).
- All factors **clipped** so `IV` stays in a sane band (e.g. `[0.02, 3.0]`).

Default parameters are equity-index-typical and live in a small dataclass so a
strategy can override them. A `realistic_surface(prices, **overrides)` helper
constructs a `ParametricSurface` (computing `atm_vol` from `prices`) in one line.

### 2. Spread model — `tradinglib/options/spread.py`

```python
class SpreadModel(Protocol):
    def half_spread_frac(self, mid: float, m: float, dte: int) -> float: ...
```

**`NoSpread`** — returns `0.0` (frictionless; backward-compat + before/after
comparison).

**`ParametricSpread`** — half-spread as a fraction of premium, widening for OTM
and short-DTE, with an absolute floor (cheap options still cost a minimum tick
to cross):

```
half_frac = clip(base + otm_penalty·|m| + short_dte_penalty/√dte,  0, max_frac)
```

The engine converts this to a per-share price offset with a `min_tick` floor
(default `$0.05`):

```
fill_buy  = max(mid·(1 + half_frac), mid + min_tick)
fill_sell = min(mid·(1 − half_frac), mid − min_tick)
```

Defaults: `base = 0.01`, `otm_penalty ≈ 0.05`, `short_dte_penalty ≈ 0.02`,
`max_frac = 0.5`, `min_tick = 0.05`. These are plausible equity-index option
spreads, tunable per model.

### 3. Engine integration — `options_engine.py`

- **`OptionsEngine.__init__`** takes `surface: VolSurface` (replacing the `vol`
  scalar) and `spread: SpreadModel`. `rate` and `fee_bps` are unchanged.
  `slippage_bps` is **retained for underlying hedge trades only** (stock spreads
  are negligible vs options); option legs use `spread` instead.
- **`_price_leg` (mid)** and **`_leg_delta`** call
  `surface.iv(spot, strike, expiry, t)` instead of `self.vol`. Greeks for the
  delta-hedge use the same surface IV. (Known carry-over limitation from phase 1:
  American legs use BSM delta — unchanged here.)
- **`add_leg`** fills at the **ask** (`fill_buy` from §2: `max(mid·(1+half_frac),
  mid + min_tick)`). **`close_all_options`** and the expiry-roll sell at the
  **bid** (`fill_sell`). Commissions (`fee_bps`) apply on top of the spread, on
  traded notional.
- **`hedge_to_delta`** keeps the existing flat `slippage_bps` on the underlying.
- `_bar_notional` / turnover accounting is unchanged in shape.

### 4. Migration (backward-compatible)

Same pattern SP1 used for the `execution_prices → open_prices/fill` migration:

- `run_options_backtest(prices, strategy, *, surface=None, spread=None, vol=None, ...)`.
- `vol=` becomes a **deprecated alias**: if passed, the engine builds
  `FlatSurface(vol)` + `NoSpread` and emits a `DeprecationWarning`. This keeps the
  existing seed model and **all** current options tests **bit-identical**.
- Passing both `vol=` and `surface=` raises `ValueError`.
- Realistic stack is one line: `surface=realistic_surface(prices)`,
  `spread=ParametricSpread()`.

### 5. Demo model — `models/options/02-directional-call-spy`

One demonstrator (not four), exercising skew + term + spread on a directional
2-month trade:

- Buy a ~2-month SPY call — **ATM and an OTM (~5%) variant** — hold and roll at
  expiry.
- Run it **twice**: frictionless (`FlatSurface`+`NoSpread`) vs realistic
  (`realistic_surface`+`ParametricSpread`).
- Emit both equity curves + metrics side by side, a JSON summary, and a plot.
  The headline number is the realistic-vs-frictionless P&L gap — *the cost of
  paying real skew and spread.*

## Out of scope (this cycle)

- **Paid historical options-chain loader (SP3 proper):** ORATS / Polygon / CBOE.
  The modular `VolSurface` / `SpreadModel` interfaces make this a later drop-in.
- **Surface-aware Monte Carlo** (`simulate.py` stays constant-vol GBM). Forward
  simulation gains little from the historical surface; revisit if needed.
- **Regime-dependent skew/term (Tier 3):** skew steepening / term inversion in
  stress. Deferred — hard to validate without real quotes; add if a backtest
  looks suspiciously clean.
- **Delta/IV-space skew** (we use log-moneyness, not option delta) and
  **forward-moneyness** (`log(K/F)` vs `log(K/S)`). Minor refinements.
- Retrofitting the existing seed model onto the realistic surface (it stays a
  clean constant-vol vol-story exhibit); regenerating `MODELS.md`.

## Acceptance criteria / verification

No real quotes ⇒ no ground-truth precision target. Tests assert **qualitative +
economic** correctness (TDD, failing-test-first, matching the SP1 task style):

1. **Surface skew:** `iv(OTM put) > iv(ATM) > iv(OTM call)` at fixed `dte`.
2. **Surface calibration:** `atm_vol(t)` tracks its realized-vol input; raising
   the input vol raises queried IV monotonically.
3. **Surface term:** long-dated skew is flatter than short-dated (slope `|b_eff|`
   decreases with `dte`); IV stays within the clip band.
4. **Spread shape:** `ask > mid > bid`; `half_frac` wider for OTM and shorter
   DTE; `min_tick` floor respected on cheap options.
5. **Engine economics (central test):** open-then-immediately-close round-trip
   loses ≈ `2·half_spread + fees`; frictions monotonically reduce P&L vs
   frictionless.
6. **Backward-compat (golden):** `vol=`/`FlatSurface`+`NoSpread` reproduces the
   current seed model's metrics **bit-identically**; the deprecated `vol=` path
   warns and matches the explicit `FlatSurface` path.
7. **Demo:** the realistic run's equity is **below** the frictionless run's for
   the same signal (frictions cost money), and both produce a valid
   `BacktestResult`.
8. **Green gate:** full `pytest -q`, `ruff check`, and `mypy` on the new modules
   pass.
