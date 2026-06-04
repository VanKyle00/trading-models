# Options Backtest & Simulation — Design

**Date:** 2026-06-04
**Status:** Approved (phase 1)
**Approach:** New `tradinglib/options/` subpackage + a dedicated options backtest
engine that reuses the existing metrics core (Approach A).

## Goal

Add options trading backtest and Monte Carlo simulation capability to the
repo. Phase 1 builds a self-contained pricing/simulation core and ships one
end-to-end demonstrator model wired into the Streamlit GUI. A later phase
layers a historical options-chain loader on top of the proven engine.

The current backtest engine is single-asset and linear: PnL is
`position × underlying_return`. Options break that abstraction — payoffs are
nonlinear, positions have strikes/expiries/multiple legs, and PnL is the
*change in option value*. This design adds the machinery to handle that while
keeping the existing metrics/result/model conventions intact so options models
remain comparable to everything else in `MODELS.md`.

## Scope (phase 1)

In scope:
- Black-Scholes-Merton pricing + Greeks (European), CRR binomial tree
  (American early exercise), implied-vol inversion.
- Multi-leg portfolio position model (arbitrary calls/puts, long/short,
  multiple strikes & expiries; multi-underlying-ready, single-underlying in
  practice).
- An options backtest engine that marks a position to market over a price
  path (real or simulated), handles delta-hedging and expiry-roll, and emits a
  standard `BacktestResult`.
- GBM Monte Carlo path simulation producing a distribution of P&L outcomes.
- One seed model: delta-hedged long option on SPY, run both as a historical
  backtest and a Monte Carlo outcome simulation.
- Streamlit GUI integration for the seed model.

Out of scope (later phases):
- Historical options-chain data loader (ORATS / Polygon options / CBOE).
- Stochastic-vol models (Heston / vol surface).
- Portfolio-of-underlyings cross-asset machinery.

## Architecture

New package `tradinglib/options/`:

| Module | Purpose | Depends on |
| --- | --- | --- |
| `instruments.py` | Position data model — `OptionLeg`, `Position`, payoff/value | numpy, pandas |
| `pricing.py` | BSM price + Greeks (European); CRR binomial tree (American); implied vol | numpy, scipy.stats |
| `simulate.py` | GBM Monte Carlo path generator (vectorized, memory-bounded) | numpy |

New engine module `tradinglib/backtest/options_engine.py`, alongside the
existing `engine.py` (vectorized) and `event_engine.py`.

### Instrument model (`instruments.py`)

```python
@dataclass(frozen=True)
class OptionLeg:
    right: Literal["call", "put"]
    strike: float
    expiry: pd.Timestamp
    quantity: float          # +long / -short, in contracts
    style: Literal["european", "american"] = "european"
    underlying: str = "SPY"  # multi-underlying-ready, single by default

@dataclass
class Position:
    legs: list[OptionLeg]
    shares: float = 0.0      # underlying shares held (hedging / covered calls)
    cash: float = 0.0
```

- A `Position` is the unit the engine marks to market.
  `value(spot, t, vol, rate)` sums each leg's priced value ×
  quantity × contract-multiplier (100), plus `shares × spot + cash`.
- Supports every multi-leg structure (spreads, straddles, condors, covered
  calls) and is multi-underlying-ready without building portfolio machinery now.

### Pricing (`pricing.py`)

```python
def bs_price(right, spot, strike, t, vol, rate, div=0.0) -> float
def bs_greeks(right, spot, strike, t, vol, rate, div=0.0) -> Greeks  # delta, gamma, vega, theta, rho
def crr_price(right, spot, strike, t, vol, rate, style, steps=512, div=0.0) -> float
def implied_vol(price, right, spot, strike, t, rate, div=0.0) -> float  # Newton/Brent inversion
```

- Pure functions, vectorizable over arrays (keeps Monte Carlo repricing fast).
- European → closed-form BSM; American → CRR tree with early-exercise check.
- `Greeks` is a small frozen dataclass so the engine reads `.delta` for hedging.
- `implied_vol` supports the realized-vs-implied story and the future
  historical-chain phase.

### Options backtest engine (`options_engine.py`)

The engine computes its own equity curve by **mark-to-market**, not through
`run_backtest`'s `position × return` math (the linear assumption we are
escaping). It then calls the existing `compute_metrics` so the model stays
comparable.

Per-bar flow over a price path:
1. Reprice the live `Position` at the new spot/time (vol & rate from
   assumptions or an IV input).
2. If delta-hedging is requested, trade underlying shares to flatten net
   delta; record the trade as turnover.
3. At a leg's expiry, settle intrinsic value and roll per the strategy's rule.
4. Record portfolio equity = position mark-to-market.

**Result contract** — returns a standard `BacktestResult` with fields
reinterpreted for options:

| Field | Meaning for options |
| --- | --- |
| `equity_curve` | Portfolio mark-to-market over time |
| `returns` | `equity_curve.pct_change()` |
| `position` | Net portfolio delta as a fraction of equity |
| `turnover` | Per-bar underlying + option notional traded ÷ equity |
| `metrics` | Existing `compute_metrics(returns, equity_curve, …)` |
| `config` | Vol/rate assumptions, hedge cadence, costs |

This reinterpretation is documented in `docs/methodology.md` so it is not a
silent surprise.

### Monte Carlo simulation (`simulate.py`)

```python
def gbm_paths(spot, vol, rate, days, n_paths, steps_per_day=1, dtype=float32) -> np.ndarray
def run_simulation(position_builder, sim_config) -> SimulationResult
```

- Vectorized GBM, `float32`, memory-bounded: ~10k paths ≈ 10 MB. Aggregate to
  per-path terminal/realized P&L; never retain full per-leg histories, so this
  stays well under the ~1 GB Streamlit Community Cloud cap. The GUI caps
  `n_paths` and logs if it truncates.
- New `SimulationResult` dataclass (distinct from `BacktestResult` — a
  distribution, not a single curve): `pnl_distribution`, `percentiles`
  (5/25/50/75/95), `prob_of_profit`, `expected_shortfall`, `mean`, `std`, plus
  a few sample paths for plotting.

## Seed model: `models/options/01-delta-hedged-long-option-spy/`

New `options/` family (the 5th, alongside classical / ml / microstructure /
alt-data). Standard layout: `backtest.py`, `model.md` (frontmatter →
`MODELS.md`), `notebook.ipynb`, `results/`.

- **Strategy:** buy a 1-month ATM SPY option, delta-hedge with the underlying
  each bar, roll at expiry. Prices each bar off a constant implied-vol
  assumption; P&L is realized-vs-implied vol — the clean test of pricing +
  Greeks correctness.
- **European** exercise for the clean vol story; the **American** CRR pricer is
  exercised in the notebook + tests so the feature is real and validated
  without muddying the demonstrator.
- `backtest.py` exposes `main()` (writes `results/`) and `run_for_gui(...)`
  matching the existing GUI contract, and produces both the historical-path
  backtest and the Monte Carlo distribution.

## GUI integration

Wire the model into `app/streamlit_app.py` like the existing models:
parameter controls (strike offset, tenor, IV assumption, hedge cadence,
`n_paths`), a payoff/Greeks plot, the equity curve, and a Monte Carlo P&L
histogram with percentile markers.

## Testing

- `pricing.py`: BS values vs published reference values / QuantLib in tests
  only (no runtime dependency); put-call parity; Greeks via finite-difference;
  CRR → BS convergence as steps ↑; American ≥ European.
- `simulate.py`: GBM mean/variance converge to analytic; seeded RNG for
  determinism.
- `options_engine.py`: known covered-call payoff at expiry; delta-hedged
  position is ~insensitive to small spot moves; expiry-roll continuity.
- Docs: update `docs/methodology.md` (result reinterpretation) and `README.md`
  repo tour (new family).

## Future phases

1. Historical options-chain loader + chain-based backtest on the same engine.
2. Stochastic-vol simulation (Heston / vol surface).
3. Additional seed models (covered call, short straddle as a negative-result
   tail-risk story).
