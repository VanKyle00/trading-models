# Validation & Overfitting Harness — Design Spec

**Date:** 2026-06-07
**Status:** Approved (pending written-spec review)
**Scope:** Sub-project **SP1** of a three-part effort to make model testing
trustworthy. **SP2** (realistic frictions: market-impact slippage, borrow /
financing costs) and **SP3** (historical options-chain loader + real-chain
engine integration) are deferred to their own design cycles.

## Motivation

The engine guards against look-ahead and applies linear costs, but a model can
run on perfect data and still be hopelessly overfit. The two levers that decide
whether a backtest is *trustworthy* rather than *curve-fit* are missing or
inert:

1. **Out-of-sample validation is a single chronological split.** A model is
   judged on one train/test cut (the ML model's `split_index`; rule models are
   not split at all). One split overstates out-of-sample performance and says
   nothing about whether the chosen parameters would have been knowable in time.
   Walk-forward / rolling re-optimization is the honest test, and it is
   roadmap-only today.
2. **The overfitting penalty never fires.** `compute_metrics` already implements
   the Deflated Sharpe Ratio (Bailey & López de Prado 2014, `metrics.py:74-117`)
   and deflates correctly when `n_trials > 1` — but every model calls it with
   `n_trials=1`, so the Deflated Sharpe collapses to the Probabilistic Sharpe and
   corrects for nothing. The selection bias from trying many configurations is
   never charged.

A third, smaller realism gap is corrected here because it belongs to the same
"trust the number" theme:

3. **The vectorized engine's default fill is optimistic.** `run_backtest`
   defaults to `execution_prices=None` ⇒ close-to-close fills at the *decision
   close* — you transact at the very price that produced the signal. Next-open
   fills exist but are opt-in; the default should be the conservative one.

## Decisions (locked during brainstorming)

- **Blast radius: harness + 2 demo models.** Build the `tradinglib/validation/`
  package and wire **SMA** (parameter search + sensitivity) and **XGBoost**
  (walk-forward re-fit) through it as end-to-end proofs. A documented `SignalFn`
  contract is the adoption path for the other three models later. `MODELS.md` is
  **not** regenerated this cycle; deflated numbers live in per-model validation
  artifacts.
- **Walk-forward re-optimizes each window** (true walk-forward optimization):
  re-select parameters (rule models) / re-fit (ML) on every in-sample window. The
  Deflated Sharpe counts the parameter grid as the trial count.
- **Both window modes provided; anchored/expanding is the default.** Rolling
  (fixed-size) is available via a parameter.
- **Next-open becomes the strict default (option B1).** `run_backtest` grows a
  `fill` parameter defaulting to `"next_open"`, which *requires* an opens series
  and raises otherwise. All in-repo callers and affected tests are updated.
  `execution_prices=` is retained as a deprecated alias. **This is a deliberate
  breaking change** for any caller that relied on the silent close-to-close
  default.

## Component 1 — Package layout (`tradinglib/validation/`)

```
tradinglib/validation/
  __init__.py        # public API + the SignalFn type alias
  splits.py          # rolling_windows(), anchored_windows()
  search.py          # grid_search() -> best params + true n_trials
  walk_forward.py    # the harness: per-window select -> apply -> stitch -> deflate
  sensitivity.py     # parameter_sensitivity() + metrics_by_regime()
```

A new package (not `tradinglib/eval/`, which holds trade analytics) keeps the
validation methodology cohesive and independently testable.

## Component 2 — The model↔harness contract (`SignalFn`)

A single callable spans both model shapes; no per-model classes are required.

```python
SignalFn = Callable[[pd.DataFrame, pd.DataFrame, dict], pd.Series]
# make_signal(train_df, test_df, params) -> target-position signal, indexed exactly like test_df
```

- `train_df` — all bars up to and including the in-sample boundary.
- `test_df` — the out-of-sample window.
- `params` — one configuration drawn from the search grid.
- **Return** — a target-position series indexed *identically* to `test_df`
  (the harness asserts this).

Adapters for the two demos:

- **Rule-based (SMA):** ignores fitting; computes `build_signal` over
  `concat(train_df.tail(lookback), test_df)` so rolling indicators are warm at
  the boundary, then returns the `test_df` slice.
- **ML (XGBoost):** fits `train_model` on `train_df` features using `params` as
  hyperparameters, predicts on `test_df`, maps prediction → position.

## Component 3 — Splits (`splits.py`)

```python
anchored_windows(index, initial_train, test_size, step=None) -> list[tuple[Index, Index]]
rolling_windows(index, train_size, test_size, step=None)     -> list[tuple[Index, Index]]
```

- Sizes are in **bar counts** (calendar-sized windows are out of scope, noted).
- `step` defaults to `test_size` ⇒ **non-overlapping** out-of-sample segments
  that tile the evaluable span.
- **anchored:** train start fixed, train window grows, test slides forward.
- **rolling:** fixed-size train window slides forward.
- Each yielded pair is `(train_index, test_index)` with **disjoint** indices.

## Component 4 — Grid search (`search.py`)

```python
grid_search(param_grid: dict[str, list], score_fn: Callable[[dict], float]) -> SearchResult
# SearchResult: best_params, best_score, n_trials, results: list[(params, score)]
```

- Evaluates the full Cartesian product of `param_grid`.
- `n_trials == len(product)` — the honest count of distinct hypotheses tried.
- Deterministic ordering; ties broken by first-seen.
- Used standalone (search once, report deflated) and inside `walk_forward` for
  per-window selection (`score_fn` = in-sample-window backtest Sharpe).

## Component 5 — Walk-forward harness (`walk_forward.py`)

```python
walk_forward(
    data, make_signal, *,
    param_grid,
    mode="anchored",                 # or "rolling"
    initial_train, test_size, step=None,
    price_col="close", open_col="open",
    fee_bps=1.0, slippage_bps=0.5, periods_per_year=252,
) -> WalkForwardResult
```

**Per window:** `grid_search` selects the best params by an **in-sample-window**
backtest Sharpe (run `make_signal(train, train, p)` → `run_backtest` on the train
window) → `make_signal(train, test, best)` produces the out-of-sample slice.

**After all windows:** concatenate the per-window OOS signal slices into one
continuous series spanning the union of test windows, run the engine **once**
over that stitched OOS span with **next-open fills**, and report metrics
**deflated by `n_trials = len(param_grid product)`**. Parameter changes at window
boundaries produce real turnover (a re-balance), which is charged.

`WalkForwardResult` carries:

- `oos_result: BacktestResult` — the stitched, deflated out-of-sample run.
- `windows: pd.DataFrame` — one row per window: train/test date bounds, chosen
  params, in-sample score, OOS Sharpe.
- `param_stability: dict` — how often each parameter changed window-to-window
  (frequent jumps are themselves an overfit smell).

**`n_trials` choice (documented assumption):** the grid size is used as the
trial count. Re-optimizing per window makes a single "true" count ambiguous;
grid size is the conservative, defensible number of independent hypotheses the
designer chose among.

## Component 6 — Sensitivity & regime diagnostics (`sensitivity.py`)

```python
parameter_sensitivity(data, make_signal, param_grid, *, train, test, ...) -> pd.DataFrame
# tidy frame: one row per grid config -> its OOS metrics
```

Evaluates **every** grid config on a fixed OOS window so the chosen config can be
read against its neighbours: a **plateau** of comparable scores is robust; a
lonely **spike** is overfit. A small helper summarizes plateau-ness (fraction of
configs within X% of the best).

```python
metrics_by_regime(returns, equity_curve, *, by="year" | labels: pd.Series,
                  periods_per_year=252) -> pd.DataFrame
# compute_metrics per sub-period -> one row per regime
```

Slices by calendar year or a supplied regime-label Series; a convenience
`vol_regime(prices, window, n_bins)` labeler is included. Reveals whether an
edge is concentrated in a single regime.

## Component 7 — Engine change: next-open default (`tradinglib/backtest/engine.py`)

### New signature

```python
run_backtest(
    prices, signals, *,
    open_prices=None,
    fill="next_open",                # or "decision_close"
    execution_prices=None,           # deprecated alias for open_prices
    initial_capital=100_000.0,
    fee_bps=1.0, slippage_bps=0.5,
    periods_per_year=252, n_trials=1,
)
```

- `fill="next_open"` (**default**) **requires** `open_prices` (same index as
  `prices`); raises `ValueError` if missing. Fill convention is the existing
  next-open mechanic (`engine.py:101-109`): the entry bar earns
  `open[t] → close[t]`, held bars stay close-to-close, costs charged on the fill
  bar.
- `fill="decision_close"` ignores opens and reproduces today's close-to-close
  behavior **bit-identically** (explicit opt-in).
- `execution_prices=` is accepted as a **deprecated alias** for `open_prices`
  (emits `DeprecationWarning`, implies `fill="next_open"`).

### Callers updated

- `models/classical/01-sma-crossover-spy/backtest.py` and
  `models/ml/01-gbm-next-day-return-spy/backtest.py` already pass opens —
  migrate `execution_prices=` → `open_prices=`.
- `event_engine.py` (`run_event_backtest`) builds an opens series from `Bar.open`
  and passes `open_prices=`, `fill="next_open"`.
- **Tests** that call `run_backtest` with closes only now error under the new
  default; each is updated to pass opens or set `fill="decision_close"`.

### Out-of-scope for this component

`run_options_backtest` fill is **unchanged** — the options engine's next-open
asymmetry is already documented as intentional in the backtest-accuracy spec.

## Component 8 — Demos (proof, written to each model's `results/`)

- **SMA:** anchored walk-forward re-optimizing `fast × slow`, plus a sensitivity
  sweep. Writes `results/walk_forward.json` (per-window table + deflated OOS
  metrics) and a stitched-equity plot. Reports **deflated OOS Sharpe vs the old
  single-split Sharpe** — the honesty delta.
- **XGBoost:** anchored walk-forward **re-fitting** per window over a small
  hyperparameter grid (e.g. `max_depth`, `n_estimators`), seeded for
  determinism. Writes the same artifacts.

## Testing (TDD)

1. **Splits:** anchored grows / rolling stays fixed; train and test indices are
   disjoint; non-overlapping test segments tile the evaluable span.
2. **Grid search:** evaluates exactly `len(product)` configs; `n_trials` equals
   that count; selects the max score deterministically.
3. **Walk-forward selection:** on synthetic data with a *known* best parameter,
   the harness selects it; stitched OOS index equals the union of test windows;
   `n_trials` propagated to `compute_metrics`.
4. **No look-ahead:** the OOS signal index equals `test_df`'s index (never
   borrows future test bars into selection).
5. **Sensitivity / regime:** sweep frame has one row per config; regime
   breakdown partitions the return series with no overlap or loss.
6. **Engine next-open:** `fill="next_open"` without opens raises; with opens is
   **bit-identical** to the prior `execution_prices` path; `fill="decision_close"`
   matches the old close-to-close golden snapshot; the `execution_prices` alias
   warns and matches.

## Out of scope (this cycle)

- SP2 (market-impact / capacity slippage, borrow / financing costs) and SP3
  (historical options loader + real-chain integration).
- Retrofitting the other three models onto the harness; regenerating `MODELS.md`.
- Purged / embargoed cross-validation (López de Prado). Our test windows are
  sequential and non-overlapping, so leakage is limited; purging around the
  train/test boundary for multi-bar labels is a future refinement.
- Calendar-sized windows (we use bar counts).
- Nested / combinatorial CV for selection; bootstrap confidence intervals.
