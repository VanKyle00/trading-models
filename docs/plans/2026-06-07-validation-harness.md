# Validation & Overfitting Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `tradinglib/validation/` walk-forward + grid-search + sensitivity/regime harness that makes the existing Deflated Sharpe penalty actually fire, and make next-open fills the engine's default — so backtests can be trusted as robust rather than curve-fit.

**Architecture:** A new `tradinglib/validation/` package with four focused modules (`splits`, `search`, `walk_forward`, `sensitivity`) sharing the existing `run_backtest` / `compute_metrics` core. Models plug in via one callable, `SignalFn(train_df, test_df, params) -> signal`. The vectorized engine gains a `fill="next_open"|"decision_close"` parameter defaulting to `next_open`. Two existing models (SMA, XGBoost) are wired through the harness as proofs.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, xgboost, pytest (+ hypothesis), ruff, mypy. Spec: `docs/specs/2026-06-07-validation-harness-design.md`.

---

## Task 1: Engine — next-open default + docstring fix

**Files:**
- Modify: `tradinglib/backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_backtest_engine.py`)

```python
def test_next_open_is_default_and_requires_opens(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    with pytest.raises(ValueError, match="open_prices"):
        run_backtest(rising_prices, signals)


def test_decision_close_matches_legacy_closetoclose(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fill="decision_close", fee_bps=0, slippage_bps=0)
    # Full-size long on a 1%/bar uptrend, close-to-close: equity must grow.
    assert result.equity_curve.iloc[-1] > 150_000.0
    assert result.config["execution"] == "decision_close"


def test_open_prices_next_open_entry_bar() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 110.0, 121.0], index=idx)
    opens = pd.Series([100.0, 105.0, 121.0], index=idx)
    signals = pd.Series([1.0, 1.0, 1.0], index=idx)
    res = run_backtest(close, signals, open_prices=opens, fee_bps=0, slippage_bps=0)
    assert res.returns.iloc[1] == pytest.approx(110.0 / 105.0 - 1.0)
    assert res.config["execution"] == "next_open"


def test_execution_prices_alias_warns_and_matches() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 110.0, 121.0], index=idx)
    opens = pd.Series([100.0, 105.0, 121.0], index=idx)
    signals = pd.Series([1.0, 1.0, 1.0], index=idx)
    with pytest.warns(DeprecationWarning, match="execution_prices"):
        legacy = run_backtest(close, signals, execution_prices=opens, fee_bps=0, slippage_bps=0)
    explicit = run_backtest(close, signals, open_prices=opens, fee_bps=0, slippage_bps=0)
    pd.testing.assert_series_equal(legacy.equity_curve, explicit.equity_curve)


def test_open_prices_index_mismatch_raises(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    bad = pd.Series(1.0, index=pd.date_range("2099-01-01", periods=len(rising_prices), freq="D"))
    with pytest.raises(ValueError, match="open_prices"):
        run_backtest(rising_prices, signals, open_prices=bad)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_backtest_engine.py -k "next_open or decision_close or open_prices or alias" -v`
Expected: FAIL (e.g. current default has no opens requirement; `fill`/`open_prices` kwargs unknown).

- [ ] **Step 3: Rewrite `run_backtest` and the module docstring**

Replace the module docstring (`engine.py:1-22`) with:

```python
"""Vectorized backtest engine for bar-level strategies.

Given a price series and a target-position series, computes the equity curve,
per-bar net returns, and standard performance metrics under linear
transaction-cost assumptions.

Conventions
-----------
- Bars are equally spaced; the engine does not try to handle calendar gaps.
- A signal at bar ``t`` is lagged one bar before multiplying by returns, which
  prevents look-ahead bias. By default trades fill at the **next bar's open**
  (``fill="next_open"``, which requires ``open_prices``): the entry bar earns
  ``open[t] -> close[t]`` and the overnight gap is not captured. Pass
  ``fill="decision_close"`` to fill at the decision bar's own close (optimistic;
  close-to-close).
- A position of ``1.0`` means fully invested; ``-1.0`` fully short; ``0.0`` flat.
  Fractional values size partially.
- Transaction costs are linear in turnover, in basis points (``1 bp = 0.01%``).
  Slippage is symmetric.

An event-driven front-end (:mod:`tradinglib.backtest.event_engine`) generates
signals via a per-bar callback and feeds them through this same vectorized core,
so PnL and metrics are identical.
"""
```

Add `import warnings` (after `from __future__ import annotations`, before the dataclass import block). Replace the `run_backtest` signature and the top of its body:

```python
def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
    n_trials: int = 1,
    open_prices: pd.Series | None = None,
    fill: str = "next_open",
    execution_prices: pd.Series | None = None,
) -> BacktestResult:
```

Replace the docstring's `execution_prices` paragraph (`engine.py:77-81`) with:

```python
    open_prices:
        Per-bar open prices (same index as ``prices``). Required when
        ``fill="next_open"``: a position change is filled at this bar's open, so
        the entry bar earns ``open -> close``. Held bars stay close-to-close.
    fill:
        ``"next_open"`` (default) or ``"decision_close"``. ``"decision_close"``
        fills at the decision bar's own close (the prior, optimistic behavior).
    execution_prices:
        Deprecated alias for ``open_prices`` (implies ``fill="next_open"``).
```

Immediately after the existing docstring, before `if not prices.index.equals(...)`, insert the resolution block:

```python
    if execution_prices is not None:
        warnings.warn(
            "execution_prices= is deprecated; pass open_prices= instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if open_prices is None:
            open_prices = execution_prices
        fill = "next_open"

    if fill not in ("next_open", "decision_close"):
        raise ValueError(f"fill must be 'next_open' or 'decision_close', got {fill!r}")
    if fill == "next_open" and open_prices is None:
        raise ValueError(
            "fill='next_open' (the default) requires open_prices; pass the open "
            "series, or set fill='decision_close' for close-to-close fills"
        )
```

Change the `execution_prices` index check (`engine.py:88-89`) to:

```python
    if open_prices is not None and not prices.index.equals(open_prices.index):
        raise ValueError("prices and open_prices must share the same index")
```

Change the fill branch (`engine.py:101-109`) from `if execution_prices is not None:` to:

```python
    if fill == "next_open":
        # A position change in force at bar t was decided at close[t-1] and is
        # filled at bar t's OPEN. The entry bar earns open[t] -> close[t]; held
        # bars stay close-to-close. Removes the optimism of filling at the very
        # close used to make the decision.
        entered = turnover > 0.0
        entry_returns = (prices / open_prices - 1.0).fillna(0.0)
        price_returns = price_returns.where(~entered, entry_returns)
```

Change the config line (`engine.py:135`) to:

```python
            "execution": fill,
```

- [ ] **Step 4: Run to verify the new tests pass**

Run: `uv run pytest tests/test_backtest_engine.py -k "next_open or decision_close or open_prices or alias" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat(engine): make next-open fills the default (fill=); deprecate execution_prices"
```

---

## Task 2: Migrate callers + existing engine tests to the new fill API

**Files:**
- Modify: `tradinglib/backtest/event_engine.py:159-163`
- Modify: `models/classical/01-sma-crossover-spy/backtest.py:65-72`
- Modify: `models/ml/01-gbm-next-day-return-spy/backtest.py:96-103, 137-143`
- Modify: `models/alt-data/01-google-trends-btc/backtest.py:109-117`
- Modify: `tests/test_backtest_engine.py` (existing close-only tests)

- [ ] **Step 1: Migrate the four real callers (`execution_prices=` → `open_prices=`)**

In `event_engine.py`, the `return run_backtest(...)` (around line 159) — change `execution_prices=open_prices,` to:

```python
        open_prices=open_prices,
        fill="next_open",
```

In `models/classical/01-sma-crossover-spy/backtest.py` (~line 68) change `execution_prices=bars["open"],` to `open_prices=bars["open"],`.

In `models/ml/01-gbm-next-day-return-spy/backtest.py` change **both** call sites (`execution_prices=user_opens,` ~line 99 and `execution_prices=oos_opens,` ~line 140) to `open_prices=user_opens,` and `open_prices=oos_opens,` respectively.

In `models/alt-data/01-google-trends-btc/backtest.py` (~line 112) change `execution_prices=weekly_open,` to `open_prices=weekly_open,`.

- [ ] **Step 2: Migrate the close-only tests in `tests/test_backtest_engine.py`**

These tests intentionally exercise core math without opens; add `fill="decision_close"` so they stay bit-identical. Edit each call:

- `test_returns_result_object` (line 29): `run_backtest(rising_prices, signals, fill="decision_close", fee_bps=0, slippage_bps=0)`
- `test_long_position_captures_uptrend` (37): add `fill="decision_close"`
- `test_zero_position_means_flat` (44): add `fill="decision_close"`
- `test_short_position_inverts_uptrend` (50): add `fill="decision_close"`
- `test_flat_prices_yield_flat_equity` (56): add `fill="decision_close"`
- `test_costs_drag_on_active_strategy` (67-68): add `fill="decision_close"` to both calls
- `test_signal_is_lagged_no_lookahead` (77): add `fill="decision_close"`
- `test_metrics_keys_present` (98): `run_backtest(rising_prices, signals, fill="decision_close")`
- `test_max_drawdown_is_nonpositive` (112): add `fill="decision_close"`
- `test_n_trials_lowers_deflated_sharpe` (122-123): add `fill="decision_close"` to both calls

For `test_mismatched_index_raises` (line 85) and `test_short_series_raises` (line 93): these must still raise their *own* errors before the fill check. Add `fill="decision_close"` to both so the fill check doesn't pre-empt them:
`run_backtest(a, b, fill="decision_close")` and `run_backtest(prices, signals, fill="decision_close")`.

- [ ] **Step 3: Replace the three obsolete `execution_prices` tests**

Delete `test_execution_prices_use_open_on_entry_bar`, `test_execution_prices_none_is_unchanged`, and `test_execution_prices_index_mismatch_raises` (lines 128-161) — their behavior is now covered by Task 1's new tests (`test_open_prices_next_open_entry_bar`, `test_execution_prices_alias_warns_and_matches`, `test_open_prices_index_mismatch_raises`).

- [ ] **Step 4: Run the full affected suites**

Run: `uv run pytest tests/test_backtest_engine.py tests/test_event_engine.py -v`
Expected: PASS (no failures, no DeprecationWarnings).

- [ ] **Step 5: Sanity-run two migrated models (uses cached data)**

Run: `uv run python -c "import sys; sys.path.insert(0, 'models/classical/01-sma-crossover-spy'); import backtest; print(backtest.run_for_gui()['result'].config['execution'])"`
Expected: prints `next_open`.

- [ ] **Step 6: Commit**

```bash
git add tradinglib/backtest/event_engine.py models/ tests/test_backtest_engine.py
git commit -m "refactor: migrate run_backtest callers + tests to open_prices/fill API"
```

---

## Task 3: `validation/splits.py` — walk-forward windows

**Files:**
- Create: `tradinglib/validation/__init__.py` (empty placeholder for now — one line)
- Create: `tradinglib/validation/splits.py`
- Test: `tests/test_validation_splits.py`

- [ ] **Step 1: Write the failing test** (`tests/test_validation_splits.py`)

```python
"""Tests for walk-forward window generators."""
from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.validation.splits import anchored_windows, rolling_windows


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_anchored_grows_train_and_tiles_test() -> None:
    idx = _idx(100)
    wins = anchored_windows(idx, initial_train=10, test_size=5)
    assert len(wins) == (100 - 10) // 5
    assert len(wins[0][0]) == 10 and len(wins[1][0]) == 15  # train grows
    assert all(len(test) == 5 for _, test in wins)
    # disjoint train/test, and non-overlapping consecutive test segments
    assert wins[0][0].intersection(wins[0][1]).empty
    assert wins[0][1].max() < wins[1][1].min()


def test_rolling_keeps_train_fixed() -> None:
    idx = _idx(100)
    wins = rolling_windows(idx, train_size=10, test_size=5)
    assert all(len(train) == 10 for train, _ in wins)
    assert wins[0][0].min() < wins[1][0].min()  # window slides
    assert wins[0][0].intersection(wins[0][1]).empty


def test_invalid_sizes_raise() -> None:
    idx = _idx(10)
    with pytest.raises(ValueError):
        anchored_windows(idx, initial_train=0, test_size=5)
    with pytest.raises(ValueError):
        rolling_windows(idx, train_size=5, test_size=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_validation_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: tradinglib.validation.splits`.

- [ ] **Step 3: Implement**

Create `tradinglib/validation/__init__.py` with a single line (expanded in Task 7):

```python
"""Validation & overfitting harness — walk-forward, search, sensitivity."""
```

Create `tradinglib/validation/splits.py`:

```python
"""Walk-forward window generators.

Produce ``(train_index, test_index)`` pairs over a time-ordered index. Anchored
windows grow the training span from a fixed start; rolling windows slide a
fixed-size training span. With the default ``step == test_size`` the test
segments are non-overlapping and tile the evaluable span exactly once.
"""
from __future__ import annotations

import pandas as pd


def anchored_windows(
    index: pd.Index,
    initial_train: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-train walk-forward windows."""
    if initial_train < 1 or test_size < 1:
        raise ValueError("initial_train and test_size must be >= 1")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")

    windows: list[tuple[pd.Index, pd.Index]] = []
    n = len(index)
    train_end = initial_train
    while train_end + test_size <= n:
        windows.append((index[:train_end], index[train_end : train_end + test_size]))
        train_end += step
    return windows


def rolling_windows(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
    """Fixed-size sliding-train walk-forward windows."""
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")

    windows: list[tuple[pd.Index, pd.Index]] = []
    n = len(index)
    start = 0
    while start + train_size + test_size <= n:
        s = start + train_size
        windows.append((index[start:s], index[s : s + test_size]))
        start += step
    return windows
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_validation_splits.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/validation/__init__.py tradinglib/validation/splits.py tests/test_validation_splits.py
git commit -m "feat(validation): add walk-forward split generators"
```

---

## Task 4: `validation/search.py` — grid search with trial count

**Files:**
- Create: `tradinglib/validation/search.py`
- Test: `tests/test_validation_search.py`

- [ ] **Step 1: Write the failing test** (`tests/test_validation_search.py`)

```python
"""Tests for grid search."""
from __future__ import annotations

from tradinglib.validation.search import SearchResult, expand_grid, grid_search


def test_expand_grid_cartesian_product() -> None:
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = expand_grid(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos and {"a": 2, "b": 30} in combos


def test_expand_grid_empty_is_single_empty_config() -> None:
    assert expand_grid({}) == [{}]


def test_grid_search_picks_max_and_counts_trials() -> None:
    grid = {"x": [1, 2, 3, 4]}
    res = grid_search(grid, score_fn=lambda p: -((p["x"] - 3) ** 2))  # peak at x=3
    assert isinstance(res, SearchResult)
    assert res.best_params == {"x": 3}
    assert res.n_trials == 4
    assert len(res.results) == 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_validation_search.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** (`tradinglib/validation/search.py`)

```python
"""Grid search over a parameter space with an honest trial count.

``grid_search`` evaluates every point in the Cartesian product of ``param_grid``
with a caller-supplied ``score_fn`` and returns the best configuration plus the
number of configurations tried (``n_trials``) — the value to feed the Deflated
Sharpe so the multiple-testing penalty reflects reality.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    best_params: dict
    best_score: float
    n_trials: int
    results: list[tuple[dict, float]] = field(default_factory=list)


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a param grid -> list of param dicts (deterministic)."""
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(param_grid[k] for k in keys))]


def grid_search(param_grid: dict[str, list], score_fn: Callable[[dict], float]) -> SearchResult:
    """Evaluate every config; return the highest-scoring one and the trial count."""
    configs = expand_grid(param_grid)
    results = [(cfg, float(score_fn(cfg))) for cfg in configs]
    best_params, best_score = max(results, key=lambda pair: pair[1])
    return SearchResult(best_params=best_params, best_score=best_score, n_trials=len(configs), results=results)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_validation_search.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/validation/search.py tests/test_validation_search.py
git commit -m "feat(validation): add grid search with honest trial count"
```

---

## Task 5: `validation/walk_forward.py` — the harness

**Files:**
- Create: `tradinglib/validation/walk_forward.py`
- Test: `tests/test_validation_walk_forward.py`

- [ ] **Step 1: Write the failing test** (`tests/test_validation_walk_forward.py`)

```python
"""Tests for the walk-forward harness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.validation.walk_forward import WalkForwardResult, walk_forward


def _data(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)  # steady uptrend
    return pd.DataFrame({"close": close, "open": close}, index=idx)


def _make_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.Series:
    # "long" param True => fully invested; False => flat.
    value = 1.0 if params["long"] else 0.0
    return pd.Series(value, index=test.index)


def test_walk_forward_selects_best_param_and_deflates() -> None:
    data = _data()
    grid = {"long": [True, False]}
    res = walk_forward(
        data, _make_signal, param_grid=grid, mode="anchored",
        initial_train=40, test_size=20,
    )
    assert isinstance(res, WalkForwardResult)
    # On a pure uptrend, "long" wins every window.
    assert (res.windows["param_long"] == True).all()  # noqa: E712
    # OOS index is the union of test windows; n_trials == grid size.
    assert res.oos_result.config["n_trials"] == 2
    assert res.oos_result.equity_curve.iloc[-1] > 100_000.0
    assert "long" in res.param_stability


def test_walk_forward_rejects_misindexed_signal() -> None:
    data = _data()

    def bad_signal(train, test, params):
        return pd.Series(1.0, index=test.index[:-1])  # wrong length

    with pytest.raises(ValueError, match="indexed like test"):
        walk_forward(data, bad_signal, param_grid={"long": [True]}, mode="anchored",
                     initial_train=40, test_size=20)


def test_walk_forward_requires_window_sizing() -> None:
    data = _data()
    with pytest.raises(ValueError, match="initial_train"):
        walk_forward(data, _make_signal, param_grid={"long": [True]}, mode="anchored", test_size=20)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_validation_walk_forward.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** (`tradinglib/validation/walk_forward.py`)

```python
"""Walk-forward optimization harness.

Drives a model across rolling or anchored windows. On each window it selects the
best parameters by in-sample-window backtest Sharpe, applies them to the
out-of-sample window, and finally stitches all OOS slices into one continuous
run scored with the Deflated Sharpe deflated by the parameter-grid size.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from tradinglib.backtest.engine import BacktestResult, run_backtest
from tradinglib.validation.search import expand_grid, grid_search
from tradinglib.validation.splits import anchored_windows, rolling_windows

# make_signal(train_df, test_df, params) -> target-position series indexed like test_df
SignalFn = Callable[[pd.DataFrame, pd.DataFrame, dict], pd.Series]


@dataclass
class WalkForwardResult:
    oos_result: BacktestResult
    windows: pd.DataFrame
    param_stability: dict


def _in_sample_sharpe(
    train: pd.DataFrame, make_signal: SignalFn, params: dict, *,
    price_col: str, open_col: str, fee_bps: float, slippage_bps: float, periods_per_year: int,
) -> float:
    sig = make_signal(train, train, params)
    res = run_backtest(
        train[price_col], sig, open_prices=train[open_col], fill="next_open",
        fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year,
    )
    return float(res.metrics["sharpe"])


def walk_forward(
    data: pd.DataFrame,
    make_signal: SignalFn,
    *,
    param_grid: dict[str, list],
    mode: Literal["anchored", "rolling"] = "anchored",
    test_size: int,
    initial_train: int | None = None,
    train_size: int | None = None,
    step: int | None = None,
    price_col: str = "close",
    open_col: str = "open",
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    if price_col not in data or open_col not in data:
        raise ValueError(f"data must have '{price_col}' and '{open_col}' columns")

    if mode == "anchored":
        if initial_train is None:
            raise ValueError("mode='anchored' requires initial_train")
        windows = anchored_windows(data.index, initial_train, test_size, step)
    elif mode == "rolling":
        if train_size is None:
            raise ValueError("mode='rolling' requires train_size")
        windows = rolling_windows(data.index, train_size, test_size, step)
    else:
        raise ValueError(f"mode must be 'anchored' or 'rolling', got {mode!r}")
    if not windows:
        raise ValueError("no walk-forward windows — check sizes vs data length")

    n_trials = len(expand_grid(param_grid))
    oos_signals: list[pd.Series] = []
    rows: list[dict] = []

    for train_idx, test_idx in windows:
        train = data.loc[train_idx]
        test = data.loc[test_idx]
        search = grid_search(
            param_grid,
            lambda p, _train=train: _in_sample_sharpe(
                _train, make_signal, p, price_col=price_col, open_col=open_col,
                fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year,
            ),
        )
        sig = make_signal(train, test, search.best_params)
        if not sig.index.equals(test.index):
            raise ValueError("make_signal must return a series indexed like test_df")
        oos_signals.append(sig)
        rows.append({
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "test_start": test_idx[0], "test_end": test_idx[-1],
            "in_sample_sharpe": search.best_score,
            **{f"param_{k}": v for k, v in search.best_params.items()},
        })

    oos_signal = pd.concat(oos_signals)
    if oos_signal.index.has_duplicates:
        raise ValueError("overlapping test windows (use step >= test_size)")
    oos_result = run_backtest(
        data.loc[oos_signal.index, price_col], oos_signal,
        open_prices=data.loc[oos_signal.index, open_col], fill="next_open",
        fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year,
        n_trials=n_trials,
    )
    windows_df = pd.DataFrame(rows)
    return WalkForwardResult(
        oos_result=oos_result, windows=windows_df,
        param_stability=_param_stability(windows_df, param_grid),
    )


def _param_stability(windows_df: pd.DataFrame, param_grid: dict[str, list]) -> dict:
    """Fraction of window-to-window transitions where each param changed."""
    stability: dict = {}
    n_transitions = max(len(windows_df) - 1, 1)
    for k in param_grid:
        col = f"param_{k}"
        if col in windows_df:
            changes = int((windows_df[col] != windows_df[col].shift()).iloc[1:].sum())
            stability[k] = changes / n_transitions
    return stability
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_validation_walk_forward.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/validation/walk_forward.py tests/test_validation_walk_forward.py
git commit -m "feat(validation): add walk-forward harness with per-window re-optimization"
```

---

## Task 6: `validation/sensitivity.py` — sensitivity + regime diagnostics

**Files:**
- Create: `tradinglib/validation/sensitivity.py`
- Test: `tests/test_validation_sensitivity.py`

- [ ] **Step 1: Write the failing test** (`tests/test_validation_sensitivity.py`)

```python
"""Tests for sensitivity + regime diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.validation.sensitivity import (
    metrics_by_regime,
    parameter_sensitivity,
    vol_regime,
)


def _data(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)
    return pd.DataFrame({"close": close, "open": close}, index=idx)


def _make_signal(train, test, params):
    return pd.Series(1.0 if params["long"] else 0.0, index=test.index)


def test_parameter_sensitivity_one_row_per_config() -> None:
    data = _data()
    grid = {"long": [True, False]}
    frame = parameter_sensitivity(
        data, _make_signal, grid,
        train_index=data.index[:60], test_index=data.index[60:],
    )
    assert len(frame) == 2
    assert {"long", "sharpe", "annualized_return", "max_drawdown"} <= set(frame.columns)


def test_vol_regime_labels_all_bars() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    rets = np.concatenate([rng.normal(0.0, 0.005, 60), rng.normal(0.0, 0.02, 60)])  # vol shift
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    labels = vol_regime(close, window=10, n_bins=3)
    assert labels.dropna().nunique() == 3
    assert len(labels) == len(close)


def test_metrics_by_regime_partitions_by_year() -> None:
    idx = pd.date_range("2020-06-01", periods=400, freq="D")  # spans 2020 + 2021
    returns = pd.Series(0.001, index=idx)
    equity = (1.0 + returns).cumprod() * 100_000.0
    frame = metrics_by_regime(returns, equity, by="year")
    assert set(frame.index) == {2020, 2021}
    assert frame["n_bars"].sum() == len(returns)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_validation_sensitivity.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** (`tradinglib/validation/sensitivity.py`)

```python
"""Parameter-sensitivity and regime-breakdown diagnostics.

``parameter_sensitivity`` evaluates every grid config on a fixed out-of-sample
window so a chosen config can be read against its neighbours (plateau = robust,
lonely spike = overfit). ``metrics_by_regime`` reports ``compute_metrics`` per
sub-period (calendar year or a supplied label series).
"""
from __future__ import annotations

import pandas as pd

from tradinglib.backtest.engine import run_backtest
from tradinglib.backtest.metrics import compute_metrics
from tradinglib.validation.search import expand_grid
from tradinglib.validation.walk_forward import SignalFn


def parameter_sensitivity(
    data: pd.DataFrame,
    make_signal: SignalFn,
    param_grid: dict[str, list],
    *,
    train_index: pd.Index,
    test_index: pd.Index,
    price_col: str = "close",
    open_col: str = "open",
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Evaluate every grid config on a fixed OOS window -> tidy metrics frame."""
    train = data.loc[train_index]
    test = data.loc[test_index]
    rows: list[dict] = []
    for params in expand_grid(param_grid):
        sig = make_signal(train, test, params)
        res = run_backtest(
            test[price_col], sig, open_prices=test[open_col], fill="next_open",
            fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year,
        )
        rows.append({
            **params,
            "sharpe": res.metrics["sharpe"],
            "annualized_return": res.metrics["annualized_return"],
            "max_drawdown": res.metrics["max_drawdown"],
        })
    return pd.DataFrame(rows)


def vol_regime(prices: pd.Series, window: int = 20, n_bins: int = 3) -> pd.Series:
    """Label each bar by trailing-volatility quantile bucket (vol_q0 = calmest)."""
    vol = prices.pct_change().rolling(window).std()
    labels = pd.qcut(vol, n_bins, labels=[f"vol_q{i}" for i in range(n_bins)])
    return pd.Series(labels, index=prices.index, name="regime")


def metrics_by_regime(
    returns: pd.Series,
    equity_curve: pd.Series,
    *,
    by: str | pd.Series = "year",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """compute_metrics per sub-period (calendar 'year' or a label series)."""
    if isinstance(by, pd.Series):
        labels = by.reindex(returns.index)
    elif by == "year":
        labels = pd.Series(returns.index.year, index=returns.index, name="regime")
    else:
        raise ValueError("by must be 'year' or a label Series")

    rows: list[dict] = []
    for label, grp in returns.groupby(labels):
        eq = equity_curve.loc[grp.index]
        rows.append({"regime": label, **compute_metrics(grp, eq, periods_per_year=periods_per_year)})
    return pd.DataFrame(rows).set_index("regime")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_validation_sensitivity.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/validation/sensitivity.py tests/test_validation_sensitivity.py
git commit -m "feat(validation): add parameter sensitivity + regime breakdown"
```

---

## Task 7: Public API (`validation/__init__.py`)

**Files:**
- Modify: `tradinglib/validation/__init__.py`
- Test: `tests/test_validation_api.py`

- [ ] **Step 1: Write the failing test** (`tests/test_validation_api.py`)

```python
"""The validation package exposes its public API at the top level."""
from __future__ import annotations

import tradinglib.validation as v


def test_public_api_present() -> None:
    for name in (
        "walk_forward", "WalkForwardResult", "SignalFn",
        "grid_search", "SearchResult", "expand_grid",
        "anchored_windows", "rolling_windows",
        "parameter_sensitivity", "metrics_by_regime", "vol_regime",
    ):
        assert hasattr(v, name), name
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_validation_api.py -v`
Expected: FAIL (e.g. `walk_forward` not an attribute).

- [ ] **Step 3: Implement** — replace `tradinglib/validation/__init__.py` with:

```python
"""Validation & overfitting harness — walk-forward, search, sensitivity."""
from tradinglib.validation.search import SearchResult, expand_grid, grid_search
from tradinglib.validation.sensitivity import (
    metrics_by_regime,
    parameter_sensitivity,
    vol_regime,
)
from tradinglib.validation.splits import anchored_windows, rolling_windows
from tradinglib.validation.walk_forward import SignalFn, WalkForwardResult, walk_forward

__all__ = [
    "SearchResult",
    "SignalFn",
    "WalkForwardResult",
    "anchored_windows",
    "expand_grid",
    "grid_search",
    "metrics_by_regime",
    "parameter_sensitivity",
    "rolling_windows",
    "vol_regime",
    "walk_forward",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_validation_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tradinglib/validation/__init__.py tests/test_validation_api.py
git commit -m "feat(validation): expose public API"
```

---

## Task 8: SMA demo — walk-forward + sensitivity

**Files:**
- Create: `models/classical/01-sma-crossover-spy/walk_forward.py`
- Test: `tests/test_sma_walk_forward.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sma_walk_forward.py`)

```python
"""The SMA walk-forward adapter returns a test-indexed 0/1 signal."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/classical/01-sma-crossover-spy"


def _load_module():
    sys.path.insert(0, str(MODEL_DIR))
    spec = importlib.util.spec_from_file_location("sma_walk_forward", MODEL_DIR / "walk_forward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_signal_is_test_indexed_binary() -> None:
    mod = _load_module()
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    close = pd.Series(100.0 * (1.01 ** np.arange(300)), index=idx)
    data = pd.DataFrame({"close": close, "open": close}, index=idx)
    train, test = data.iloc[:250], data.iloc[250:]
    sig = mod.make_signal(train, test, {"fast": 10, "slow": 50})
    assert sig.index.equals(test.index)
    assert set(sig.unique()) <= {0.0, 1.0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_sma_walk_forward.py -v`
Expected: FAIL (file not found / module load error).

- [ ] **Step 3: Implement** (`models/classical/01-sma-crossover-spy/walk_forward.py`)

```python
"""Walk-forward validation of the SMA crossover, re-optimizing fast/slow.

Re-selects the (fast, slow) pair on each anchored in-sample window and reports
the stitched out-of-sample equity deflated by the parameter-grid size — the
honest counterpart to the single full-sample run in ``backtest.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from backtest import END, START, SYMBOL, build_signal  # noqa: E402

from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402
from tradinglib.validation import walk_forward  # noqa: E402

RESULTS = HERE / "results"
GRID = {"fast": [10, 20, 50], "slow": [100, 150, 200]}


def make_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.Series:
    # Warm the rolling means with train history straddling the boundary. Dedupe
    # so in-sample scoring (where test == train) doesn't double the index.
    closes = pd.concat([train["close"], test["close"]])
    closes = closes[~closes.index.duplicated(keep="last")]
    sig = build_signal(closes, fast=params["fast"], slow=params["slow"])
    return sig.loc[test.index]


def main() -> None:
    bars = load_daily(SYMBOL, start=START, end=END)
    data = bars[["open", "close"]].dropna()

    wf = walk_forward(
        data, make_signal, param_grid=GRID, mode="anchored",
        initial_train=756, test_size=126, step=126,  # ~3y train, ~6mo OOS
    )

    # Naive full-sample Sharpe (the optimistic number we are correcting).
    naive_sig = build_signal(data["close"], fast=50, slow=200)
    from tradinglib.backtest import run_backtest
    naive = run_backtest(data["close"], naive_sig, open_prices=data["open"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_windows": int(len(wf.windows)),
        "n_trials": int(wf.oos_result.config["n_trials"]),
        "walk_forward_oos": wf.oos_result.metrics,
        "naive_full_sample_sharpe": naive.metrics["sharpe"],
        "param_stability": wf.param_stability,
    }
    (RESULTS / "walk_forward.json").write_text(json.dumps(summary, indent=2, default=str))
    wf.windows.to_csv(RESULTS / "walk_forward_windows.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    wf.oos_result.equity_curve.plot(ax=ax, label="SMA walk-forward (OOS)")
    ax.set_title(f"{SYMBOL} — SMA crossover, walk-forward OOS")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS / "walk_forward_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test, then the end-to-end run**

Run: `uv run pytest tests/test_sma_walk_forward.py -v`
Expected: PASS.
Run: `uv run python models/classical/01-sma-crossover-spy/walk_forward.py`
Expected: prints a JSON summary; `models/classical/01-sma-crossover-spy/results/walk_forward.json` exists with `n_trials == 9` and a `walk_forward_oos` block.

- [ ] **Step 5: Commit**

```bash
git add models/classical/01-sma-crossover-spy/walk_forward.py models/classical/01-sma-crossover-spy/results/ tests/test_sma_walk_forward.py
git commit -m "feat(models): walk-forward + sensitivity for SMA crossover"
```

---

## Task 9: XGBoost demo — walk-forward re-fit

**Files:**
- Create: `models/ml/01-gbm-next-day-return-spy/walk_forward.py`
- Test: `tests/test_ml_walk_forward.py`

- [ ] **Step 1: Write the failing test** (`tests/test_ml_walk_forward.py`)

```python
"""The XGBoost walk-forward adapter fits on train and returns a test-indexed signal."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/ml/01-gbm-next-day-return-spy"


def _load_module():
    sys.path.insert(0, str(MODEL_DIR))
    spec = importlib.util.spec_from_file_location("ml_walk_forward", MODEL_DIR / "walk_forward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_signal_fits_and_is_test_indexed() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=400, freq="D")
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, 400)), index=idx)
    data = pd.DataFrame({"close": close, "open": close}, index=idx)
    train, test = data.iloc[:300], data.iloc[300:]
    sig = mod.make_signal(train, test, {"max_depth": 3, "n_estimators": 50})
    assert sig.index.equals(test.index)
    assert set(sig.unique()) <= {0.0, 1.0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ml_walk_forward.py -v`
Expected: FAIL (file not found / module load error).

- [ ] **Step 3: Implement** (`models/ml/01-gbm-next-day-return-spy/walk_forward.py`)

```python
"""Walk-forward re-fit of the XGBoost next-day-return model.

Re-fits the regressor on each anchored in-sample window (re-selecting depth /
n_estimators), predicts the OOS window, and reports the stitched OOS equity
deflated by the hyperparameter-grid size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import END, START, SYMBOL, build_features  # noqa: E402

from tradinglib.features.technical import log_return  # noqa: E402
from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402
from tradinglib.validation import walk_forward  # noqa: E402

RESULTS = HERE / "results"
GRID = {"max_depth": [3, 4], "n_estimators": [200, 300]}
_WARMUP = 60  # bars of history needed to warm the rolling features


def make_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.Series:
    feats = build_features(train["close"])
    target = log_return(train["close"], 1).shift(-1)
    aligned = pd.concat([feats, target.rename("y")], axis=1).dropna()
    cols = [c for c in aligned.columns if c != "y"]

    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )
    model.fit(aligned[cols], aligned["y"], verbose=False)

    # Warm test features with the tail of train so rolling windows are valid.
    # Dedupe so in-sample scoring (test == train) doesn't double the index.
    warm = pd.concat([train["close"].tail(_WARMUP), test["close"]])
    warm = warm[~warm.index.duplicated(keep="last")]
    test_feats = build_features(warm).loc[test.index, cols].dropna()
    pred = pd.Series(model.predict(test_feats), index=test_feats.index)
    return (pred > 0).astype(float).reindex(test.index).fillna(0.0)


def main() -> None:
    bars = load_daily(SYMBOL, start=START, end=END)
    data = bars[["open", "close"]].dropna()

    wf = walk_forward(
        data, make_signal, param_grid=GRID, mode="anchored",
        initial_train=1008, test_size=126, step=126,  # ~4y train, ~6mo OOS
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_windows": int(len(wf.windows)),
        "n_trials": int(wf.oos_result.config["n_trials"]),
        "walk_forward_oos": wf.oos_result.metrics,
        "param_stability": wf.param_stability,
    }
    (RESULTS / "walk_forward.json").write_text(json.dumps(summary, indent=2, default=str))
    wf.windows.to_csv(RESULTS / "walk_forward_windows.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    wf.oos_result.equity_curve.plot(ax=ax, label="XGBoost walk-forward (OOS)")
    ax.set_title(f"{SYMBOL} — XGBoost, walk-forward OOS")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS / "walk_forward_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test, then the end-to-end run**

Run: `uv run pytest tests/test_ml_walk_forward.py -v`
Expected: PASS.
Run: `uv run python models/ml/01-gbm-next-day-return-spy/walk_forward.py`
Expected: prints a JSON summary; `results/walk_forward.json` exists with `n_trials == 4`.

- [ ] **Step 5: Commit**

```bash
git add models/ml/01-gbm-next-day-return-spy/walk_forward.py models/ml/01-gbm-next-day-return-spy/results/ tests/test_ml_walk_forward.py
git commit -m "feat(models): walk-forward re-fit for XGBoost next-day model"
```

---

## Task 10: Docs + full green gate

**Files:**
- Modify: `docs/methodology.md`

- [ ] **Step 1: Document the new default + harness** — append a section to `docs/methodology.md`:

```markdown
## Walk-forward validation & the next-open default

`run_backtest` now defaults to `fill="next_open"` and requires an `open_prices`
series; pass `fill="decision_close"` for the prior close-to-close behavior. The
legacy `execution_prices=` argument is a deprecated alias.

The `tradinglib.validation` package adds a walk-forward harness
(`walk_forward`), grid search with an honest trial count (`grid_search`), and
sensitivity / regime diagnostics (`parameter_sensitivity`, `metrics_by_regime`).
A model adopts it by writing one `make_signal(train, test, params)` adapter; the
harness re-optimizes parameters per window and deflates the out-of-sample Sharpe
by the grid size. See `models/classical/01-sma-crossover-spy/walk_forward.py`
and `models/ml/01-gbm-next-day-return-spy/walk_forward.py`.
```

- [ ] **Step 2: Run the full suite + linters**

Run: `uv run pytest -q`
Expected: PASS (all tests, including pre-existing).
Run: `uv run ruff check tradinglib tests && uv run mypy tradinglib/validation tradinglib/backtest/engine.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add docs/methodology.md
git commit -m "docs: document next-open default + validation harness"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** splits (Task 3) ✓ · search + n_trials (Task 4) ✓ · walk-forward re-optimize/stitch/deflate (Task 5) ✓ · sensitivity + regime (Task 6) ✓ · `SignalFn` contract (Task 5, used in 8/9) ✓ · next-open B1 + deprecated alias (Task 1) ✓ · caller/test migration (Task 2) ✓ · SMA + XGBoost demos (Tasks 8/9) ✓ · docstring quick-win (Task 1, Step 3) ✓. Out-of-scope items (SP2/SP3, MODELS.md regen, purged CV) are not implemented, as intended.
- **Naming consistency:** `walk_forward`, `WalkForwardResult`, `SignalFn`, `grid_search`/`expand_grid`/`SearchResult`, `anchored_windows`/`rolling_windows`, `parameter_sensitivity`/`metrics_by_regime`/`vol_regime`, engine `fill`/`open_prices` — used identically across tasks and the `__init__` export list.
- **Window sizes** (756/126, 1008/126) assume the cached 2010–2024 daily SPY (~3,770 bars) → ~24 / ~22 windows. If a window count is 0, the harness raises a clear error (Task 5).
