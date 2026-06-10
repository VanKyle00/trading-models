# Adding a strategy to the tournament

The tournament is a registry of `StrategyDef` objects. Adding a strategy is
four files touched (one new, two count tests, README) plus one `register` call;
the conformance harness covers correctness automatically.

---

## 1. The contract

A strategy is a frozen `StrategyDef` (defined in
`tradinglib/tournament/strategies/_core.py`):

```python
@dataclass(frozen=True)
class StrategyDef:
    key: str                  # unique slug, snake_case
    name: str                 # human-readable
    style: str                # "trend" | "breakout" | "mean_reversion" | "event" | "ml"
    description: str          # plain-English rule; powers the /models page
    param_grid: dict[str, list]
    make_signal: TournamentSignalFn
    levels: LevelsFn
```

**`make_signal(train, test, params, stance) -> pd.Series`**

Return a target-position series indexed exactly like `test`: `{0, +1}` for
`stance="long"`, `{0, -1}` for `stance="short"`. The value at bar `t` may only
use information available at the *close* of bar `t`; the engine adds its own
one-bar lag before multiplying by returns, so causal signals need no extra
shift.

**`levels(bars, params, stance) -> Levels | None`**

Turn the latest bars into tomorrow's concrete entry/stop/target. Return `None`
when the setup is not actionable tonight — the strategy simply issues no ticket
that evening. `None` from a walk-forward winner is the correct answer for "the
rule has edge on this ticker but tonight's setup conditions are not met"; it
does not mean the strategy failed.

**Registration**

One `register(StrategyDef(...))` call at module level, in the correct family
module under `tradinglib/tournament/strategies/` (one module per family:
`trend.py`, `breakout.py`, `mean_reversion.py`, `event.py`, `ml.py`). The
`__init__.py` imports every family module, so importing `STRATEGIES` from the
package is enough to trigger registration — nothing else to edit in the
package itself.

---

## 2. Rule-based recipe

`base_breakout` in `tradinglib/tournament/strategies/breakout.py` is the
canonical worked example.

**Key rules:**

- Use only causal ops: `rolling`, `shift`, `ewm` on trailing windows. Never
  `center=True`; never index forward.
- Warm indicators with `_full_history(train, test)` (deduped concat) so
  rolling windows have enough history to be meaningful on the first test bar.
  In-sample scoring passes `train` as `test`; `_full_history` handles the
  overlap.
- Use `_hold_between(entry_mask, exit_mask)` for any flip-flop
  enter-and-hold state. It returns `{0, 1}`; negate for short stance.
- Slice back to `test.index` before returning:
  `return pos.loc[test.index]`.

---

## 3. ML recipe

`ridge_momentum` in `tradinglib/tournament/strategies/ml.py` is the canonical
worked example.

**Fit inside `make_signal` on `train` only.** The walk-forward harness already
windows `train` for you; fitting inside `make_signal` means the model is
re-optimized per window automatically, including in-sample mode where
`train == test`.

**Two classic leaks — both guarded by construction:**

1. **Boundary target row.** The next-bar return of the *last train row* lives
   in the test window. Constructing the target via `pct_change().shift(-1)`
   and then `dropna()` before fitting excludes that row automatically.

2. **Normalization leak.** Standardize features with train-only mean and std.
   Computing `mean` and `std` from the fit frame (train features only) and
   then applying those scalars when predicting on `full["close"]` keeps the
   test bars unseen during normalization.

**Determinism is mandatory.** The conformance harness calls `make_signal`
twice and checks bit-exact equality. Use no unseeded RNG. `ridge_momentum`
uses a closed-form `np.linalg.solve`; it is deterministic by construction.
Any strategy that uses a stochastic solver must seed it explicitly and
document the seed.

**Degrade gracefully.** When the train window is too short to fit, return
`pd.Series(0.0, index=test.index)` from `make_signal` and `None` from
`levels`. The `_MIN_FIT_ROWS = 60` guard in `ml.py` is the pattern.

---

## 4. Optional data columns

Bars may carry feature columns beyond OHLCV (e.g., `earnings` for
event-driven strategies). A strategy that needs an absent column must return
all-flat from `make_signal` and `None` from `levels` — see `event.py`
(`_pead_signal` / `_pead_levels`). This way the strategy simply never
survives in contexts where the data is missing; it never raises.

---

## 5. The police

```bash
uv run pytest tests/test_strategy_conformance.py
```

The harness in `tests/test_strategy_conformance.py` runs **every registered
strategy × every grid combo × both stances** through:

- **Causality**: the signal at bar `t` is identical when future bars are
  truncated (catches centered windows, full-sample normalization,
  fit-on-test).
- **Value/index contract**: output is test-indexed; only `{0, +1}` / `{0, -1}`
  values appear.
- **In-sample mode**: `make_signal(train, train, params, stance)` must return
  a train-indexed series without error.
- **Determinism**: two calls with identical inputs produce bit-exact output.
- **Levels geometry**: if `levels` returns non-`None`, stop is on the loss
  side of entry and target is on the profit side; input bars are never
  mutated.

**Zero new test code is required per strategy.** Parametrization is driven by
the registry; every future strategy inherits all five checks the moment it is
registered.

Write fixture tests only to verify **firing behavior** — one test that the
strategy fires when it should, one that it stays flat when it should not.
For ML strategies the fixture must carry genuine feature variance: a perfectly
geometric trend makes every feature constant (all pct-change values equal,
all SMA-distance values equal), leaving the fit as floating-point dust and the
test outcome a coin flip across BLAS versions. Use a seeded random walk with
realistic drift and volatility instead — see
`test_ridge_momentum_long_rides_a_persistent_trend` in
`tests/test_tournament_strategies.py`.

---

## 6. The deflation tax

`n_trials` in the tournament's DSR calculation equals the **total number of
grid combinations across the entire registry** — not just the new strategy's
grid. Adding even two parameter values raises the bar for every other strategy
on every future ticker run. This is by design (CON-02): the correction must
account for the whole menu of configurations tried.

Two tests pin the counts and exist precisely to force this conversation:

- `test_registry_has_nine_strategies_with_29_total_trials` in
  `tests/test_tournament_strategies.py` — asserts the strategy set and total
  trial count.
- `test_run_tournament_real_registry_structural` in
  `tests/test_tournament_run.py` — asserts `result.n_trials == 29`.

**Both tests must be updated** when you add a strategy. Compute the new total
(`sum(len(expand_grid(s.param_grid)) for s in STRATEGIES.values())`) and edit
the two pinned numbers together. Keep grids small and intentional.

---

## 7. Checklist

- [ ] Module lives in the correct family file under
  `tradinglib/tournament/strategies/` (one family per file; new families need
  an import added to `__init__.py`).
- [ ] One `register(StrategyDef(...))` call at module level.
- [ ] `uv run pytest tests/test_strategy_conformance.py` passes green.
- [ ] At least one fires-when-it-should fixture test and one stays-flat
  fixture test in `tests/test_tournament_strategies.py`; ML fixtures use a
  seeded noisy walk, not a geometric series.
- [ ] Count-pinning tests updated: registry set, total trial count in
  `test_tournament_strategies.py`, and `n_trials` in
  `test_tournament_run.py`.
- [ ] README strategy list updated by hand (repo rule — `MODELS.md` is
  auto-generated; `README.md` is not).
- [ ] Full six-step gate passes:
  `uv run pytest -q` · `uv run ruff check .` · `uv run ruff format --check .` ·
  `uv run mypy tradinglib` · registry test · conformance test.
