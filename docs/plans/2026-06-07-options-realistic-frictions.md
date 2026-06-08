# Options Realistic Frictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a synthetic, realized-vol-anchored implied-volatility surface and a bid/ask spread model to the options backtest engine so 2-week–6-month option strategies can be tested under realistic frictions, with no paid options-chain data.

**Architecture:** Two new dependency-light modules — `tradinglib/options/surface.py` (`VolSurface` protocol, `FlatSurface`, `ParametricSurface`, `realistic_surface`) and `tradinglib/options/spread.py` (`SpreadModel` protocol, `NoSpread`, `ParametricSpread`). `OptionsEngine` swaps its single `vol` scalar for a `VolSurface` (per-strike/expiry IV) and gains a `SpreadModel` (fills cross the bid/ask). The legacy `vol=` argument becomes a deprecated alias → `FlatSurface(vol)`, keeping all current behavior bit-identical. One directional demo model proves the realism by running frictionless vs realistic.

**Tech Stack:** Python 3.12, pandas, numpy, scipy (existing pricing), pytest, ruff, mypy. Spec: `docs/specs/2026-06-07-options-realistic-frictions-design.md`. Existing pricing primitives: `tradinglib/options/pricing.py` (`bs_price`, `crr_price`, `bs_greeks`), `tradinglib/options/instruments.py` (`OptionLeg`, `CONTRACT_MULTIPLIER = 100.0`).

---

## Task 1: `options/surface.py` — synthetic vol surface

**Files:**
- Create: `tradinglib/options/surface.py`
- Test: `tests/test_options_surface.py`

- [ ] **Step 1: Write the failing test** (`tests/test_options_surface.py`)

```python
"""Tests for the synthetic implied-volatility surface."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.options.surface import (
    FlatSurface,
    ParametricSurface,
    realistic_surface,
    realized_vol,
)


def _prices(n: int = 200, vol: float = 0.2, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rets = rng.normal(0.0, vol / np.sqrt(252), n)
    return pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)


def test_flat_surface_is_constant() -> None:
    s = FlatSurface(0.18)
    t = pd.Timestamp("2024-01-01")
    assert s.iv(100.0, 90.0, t + pd.Timedelta(days=30), t) == 0.18
    assert s.iv(100.0, 110.0, t + pd.Timedelta(days=180), t) == 0.18


def test_realized_vol_recovers_input_scale() -> None:
    rv = realized_vol(_prices(n=2000, vol=0.2, seed=1), window=21).dropna()
    assert 0.12 < rv.mean() < 0.30


def test_parametric_surface_has_equity_skew() -> None:
    atm = pd.Series(0.20, index=pd.date_range("2024-01-01", periods=5, freq="B"))
    s = ParametricSurface(atm_vol=atm)
    t = atm.index[0]
    expiry = t + pd.Timedelta(days=60)
    assert s.iv(100.0, 90.0, expiry, t) > s.iv(100.0, 100.0, expiry, t) > s.iv(100.0, 110.0, expiry, t)


def test_parametric_atm_equals_input_at_reference_window() -> None:
    t0 = pd.Timestamp("2024-01-01")
    atm = pd.Series([0.15, 0.40], index=[t0, t0 + pd.Timedelta(days=10)])
    s = ParametricSurface(atm_vol=atm)
    # ATM (m=0) at the 21-day reference window: term_factor == 1, skew == 1.
    low = s.iv(100.0, 100.0, t0 + pd.Timedelta(days=21), t0)
    assert low == pytest.approx(0.15, abs=1e-9)
    high = s.iv(100.0, 100.0, t0 + pd.Timedelta(days=31), t0 + pd.Timedelta(days=10))
    assert high > low


def test_long_dated_skew_is_flatter() -> None:
    atm = pd.Series(0.20, index=pd.date_range("2024-01-01", periods=5, freq="B"))
    s = ParametricSurface(atm_vol=atm)
    t = atm.index[0]
    short = s.iv(100.0, 90.0, t + pd.Timedelta(days=30), t) - s.iv(100.0, 110.0, t + pd.Timedelta(days=30), t)
    long = s.iv(100.0, 90.0, t + pd.Timedelta(days=300), t) - s.iv(100.0, 110.0, t + pd.Timedelta(days=300), t)
    assert short > long > 0


def test_realistic_surface_atm_is_time_varying_and_clipped() -> None:
    prices = _prices(n=200, vol=0.2, seed=2)
    s = realistic_surface(prices, vrp=1.15)
    assert isinstance(s, ParametricSurface)
    t_early, t_late = prices.index[40], prices.index[-1]
    iv_early = s.iv(100.0, 100.0, t_early + pd.Timedelta(days=60), t_early)
    iv_late = s.iv(100.0, 100.0, t_late + pd.Timedelta(days=60), t_late)
    assert iv_early != iv_late
    assert 0.02 <= iv_late <= 3.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_options_surface.py -v`
Expected: FAIL with `ModuleNotFoundError: tradinglib.options.surface`.

- [ ] **Step 3: Implement** (`tradinglib/options/surface.py`)

```python
"""Synthetic implied-volatility surface for the options engine.

No real options quotes — the surface is *calibrated to the underlying's own
realized volatility* and overlaid with parametric skew and term structure. It is
a stress / plausibility model: it reproduces the *shape* of real frictions (vol
regimes, skew, term structure), not the exact IV of any contract.

IV is a separable product::

    IV(K, expiry, t) = atm_vol(t) * term_factor(dte) * skew_factor(m, dte)

with ``dte = (expiry - t).days`` and ``m = log(K / spot)`` (log-moneyness).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class VolSurface(Protocol):
    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float: ...


@dataclass(frozen=True)
class FlatSurface:
    """Constant IV everywhere — reproduces the pre-surface engine behavior."""

    vol: float

    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
        return self.vol


@dataclass(frozen=True)
class SurfaceParams:
    """Shape parameters for :class:`ParametricSurface` (equity-index-typical)."""

    skew_slope: float = -0.30      # b: <0 => OTM puts richer than OTM calls
    skew_curv: float = 0.50        # c: smile curvature (>= 0)
    skew_flatten: float = 2.0      # k: skew slope decays with tenor (per year)
    term_slope: float = 0.05       # term-structure slope per sqrt-year
    ref_window_days: int = 21      # dte at which term_factor == 1.0
    iv_floor: float = 0.02
    iv_cap: float = 3.0


def realized_vol(prices: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Trailing annualized realized vol from close-to-close log returns."""
    logret = np.log(prices / prices.shift(1))
    return logret.rolling(window).std() * math.sqrt(periods_per_year)


@dataclass
class ParametricSurface:
    """Realized-vol-anchored surface with parametric skew + term structure."""

    atm_vol: pd.Series  # time-indexed ATM vol (annualized)
    params: SurfaceParams = field(default_factory=SurfaceParams)

    def iv(self, spot: float, strike: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
        p = self.params
        atm = float(self.atm_vol.asof(t))
        if not math.isfinite(atm):
            valid = self.atm_vol.dropna()
            atm = float(valid.iloc[0]) if not valid.empty else p.iv_floor
        years = max((expiry - t).days, 0) / 365.0
        m = math.log(strike / spot) if spot > 0 and strike > 0 else 0.0

        b_eff = p.skew_slope / (1.0 + p.skew_flatten * years)
        skew = 1.0 + b_eff * m + p.skew_curv * m * m
        term = 1.0 + p.term_slope * (math.sqrt(years) - math.sqrt(p.ref_window_days / 365.0))

        iv = atm * max(term, 0.0) * max(skew, 0.0)
        return float(min(max(iv, p.iv_floor), p.iv_cap))


def realistic_surface(
    prices: pd.Series,
    *,
    window: int = 21,
    vrp: float = 1.15,
    periods_per_year: int = 252,
    params: SurfaceParams | None = None,
) -> ParametricSurface:
    """Build a :class:`ParametricSurface` with ``atm_vol = realized_vol * vrp``.

    Leading NaNs from the rolling window are back-filled so early bars are usable.
    """
    atm = realized_vol(prices, window=window, periods_per_year=periods_per_year) * vrp
    atm = atm.bfill()
    return ParametricSurface(atm_vol=atm, params=params or SurfaceParams())
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_options_surface.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/surface.py tests/test_options_surface.py
git commit -m "feat(options): synthetic realized-vol-anchored vol surface (skew + term)"
```

---

## Task 2: `options/spread.py` — bid/ask spread model

**Files:**
- Create: `tradinglib/options/spread.py`
- Test: `tests/test_options_spread.py`

- [ ] **Step 1: Write the failing test** (`tests/test_options_spread.py`)

```python
"""Tests for the synthetic bid/ask spread model."""
from __future__ import annotations

from tradinglib.options.spread import NoSpread, ParametricSpread


def test_no_spread_is_zero() -> None:
    assert NoSpread().half_spread_frac(2.0, 0.0, 30) == 0.0


def test_spread_wider_for_otm() -> None:
    s = ParametricSpread()
    assert s.half_spread_frac(2.0, -0.20, 30) > s.half_spread_frac(2.0, 0.0, 30)


def test_spread_wider_for_short_dte() -> None:
    s = ParametricSpread()
    assert s.half_spread_frac(2.0, 0.0, 5) > s.half_spread_frac(2.0, 0.0, 120)


def test_spread_is_capped() -> None:
    s = ParametricSpread(max_frac=0.5)
    assert s.half_spread_frac(2.0, -10.0, 1) == 0.5


def test_min_tick_attribute_present() -> None:
    assert ParametricSpread().min_tick > 0.0
    assert NoSpread().min_tick == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_options_spread.py -v`
Expected: FAIL with `ModuleNotFoundError: tradinglib.options.spread`.

- [ ] **Step 3: Implement** (`tradinglib/options/spread.py`)

```python
"""Synthetic bid/ask spread model for option fills.

The engine fills option legs by crossing this spread: buys at the ask, sells at
the bid. The half-spread is a fraction of premium that widens for
out-of-the-money and short-dated options, with an absolute per-share floor
(``min_tick``) because even cheap options cost a minimum to cross.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpreadModel(Protocol):
    min_tick: float

    def half_spread_frac(self, mid: float, m: float, dte: int) -> float: ...


@dataclass(frozen=True)
class NoSpread:
    """Frictionless fills (mid). Backward-compat + before/after comparison."""

    min_tick: float = 0.0

    def half_spread_frac(self, mid: float, m: float, dte: int) -> float:
        return 0.0


@dataclass(frozen=True)
class ParametricSpread:
    """Half-spread fraction widening for OTM / short-DTE, with a per-share floor."""

    base: float = 0.01
    otm_penalty: float = 0.05
    short_dte_penalty: float = 0.02
    max_frac: float = 0.5
    min_tick: float = 0.05

    def half_spread_frac(self, mid: float, m: float, dte: int) -> float:
        d = max(dte, 1)
        frac = self.base + self.otm_penalty * abs(m) + self.short_dte_penalty / math.sqrt(d)
        return min(max(frac, 0.0), self.max_frac)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_options_spread.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/spread.py tests/test_options_spread.py
git commit -m "feat(options): synthetic bid/ask spread model (OTM/short-DTE widening)"
```

---

## Task 3: Engine integration — surface + spread fills

**Files:**
- Modify: `tradinglib/backtest/options_engine.py`
- Test: `tests/test_options_engine_frictions.py` (new file — keeps the existing `test_options_engine.py` untouched until Task 4)

- [ ] **Step 1: Write the failing test** (`tests/test_options_engine_frictions.py`)

```python
"""Tests for surface- and spread-aware options fills."""
from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.options.instruments import OptionLeg
from tradinglib.options.spread import NoSpread, ParametricSpread
from tradinglib.options.surface import FlatSurface, ParametricSurface


class _DoNothing:
    def on_bar(self, engine: OptionsEngine, t, spot) -> None:
        return None


class _OpenThenClose:
    def __init__(self, expiry: pd.Timestamp) -> None:
        self.expiry = expiry
        self.step = 0

    def on_bar(self, engine: OptionsEngine, t, spot) -> None:
        if self.step == 0:
            engine.add_leg(OptionLeg("call", strike=100.0, expiry=self.expiry, quantity=1.0))
        elif self.step == 1:
            engine.close_all_options()
        self.step += 1


def test_round_trip_loses_the_spread() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.Series(100.0, index=idx)
    expiry = idx[-1] + pd.Timedelta(days=60)
    surface = FlatSurface(0.2)

    res = run_options_backtest(
        prices, _OpenThenClose(expiry), surface=surface, spread=ParametricSpread(),
        fee_bps=0, slippage_bps=0,
    )
    res0 = run_options_backtest(
        prices, _OpenThenClose(expiry), surface=surface, spread=NoSpread(),
        fee_bps=0, slippage_bps=0,
    )
    # Crossing the spread costs money; the frictionless run barely moves.
    assert res.equity_curve.iloc[-1] < res0.equity_curve.iloc[-1]
    assert res0.equity_curve.iloc[-1] == pytest.approx(100_000.0, rel=0.01)


def test_surface_skew_makes_otm_put_richer_than_otm_call() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    atm = pd.Series(0.2, index=idx)
    surface = ParametricSurface(atm_vol=atm)
    expiry = idx[-1] + pd.Timedelta(days=60)
    eng = OptionsEngine(
        surface, NoSpread(), rate=0.04, fee_bps=0, slippage_bps=0, initial_capital=100_000.0
    )
    eng.t, eng.spot = idx[0], 100.0
    # Skew is an implied-vol effect: the OTM put carries higher IV than the OTM
    # call. (Price alone wouldn't show this — lognormal drift makes the OTM call
    # richer in dollars despite its lower IV.)
    put_iv = eng._leg_iv(OptionLeg("put", strike=90.0, expiry=expiry, quantity=1.0))
    call_iv = eng._leg_iv(OptionLeg("call", strike=110.0, expiry=expiry, quantity=1.0))
    assert put_iv > call_iv


def test_vol_kwarg_is_deprecated_alias() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.Series(100.0, index=idx)
    with pytest.warns(DeprecationWarning, match="vol="):
        legacy = run_options_backtest(prices, _DoNothing(), vol=0.2, rate=0.04)
    explicit = run_options_backtest(prices, _DoNothing(), surface=FlatSurface(0.2), rate=0.04)
    pd.testing.assert_series_equal(legacy.equity_curve, explicit.equity_curve)


def test_vol_and_surface_conflict_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.Series(100.0, index=idx)
    with pytest.raises(ValueError, match="either"):
        run_options_backtest(prices, _DoNothing(), vol=0.2, surface=FlatSurface(0.2))


def test_missing_surface_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.Series(100.0, index=idx)
    with pytest.raises(ValueError, match="surface"):
        run_options_backtest(prices, _DoNothing())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_options_engine_frictions.py -v`
Expected: FAIL (`run_options_backtest` has no `surface=`/`spread=` kwargs; `OptionsEngine.__init__` still takes `vol`).

- [ ] **Step 3: Edit the imports and module docstring** (`tradinglib/backtest/options_engine.py`)

Replace the import block (lines 15-24, from `from __future__ import annotations` through the `from tradinglib.options.pricing import ...` line) with:

```python
from __future__ import annotations

import math
import warnings
from typing import Protocol

import pandas as pd

from tradinglib.backtest.engine import BacktestResult
from tradinglib.backtest.metrics import compute_metrics
from tradinglib.options.instruments import CONTRACT_MULTIPLIER, OptionLeg, Position, intrinsic_value
from tradinglib.options.pricing import bs_greeks, bs_price, crr_price
from tradinglib.options.spread import NoSpread, SpreadModel
from tradinglib.options.surface import FlatSurface, VolSurface
```

- [ ] **Step 4: Replace `OptionsEngine.__init__`**

Replace the constructor (the `def __init__(self, vol, rate, fee_bps, slippage_bps, initial_capital)` block, lines 43-57) with:

```python
    def __init__(
        self,
        surface: VolSurface,
        spread: SpreadModel,
        rate: float,
        fee_bps: float,
        slippage_bps: float,
        initial_capital: float,
    ) -> None:
        self.surface = surface
        self.spread = spread
        self.rate = rate
        self.option_fee_rate = fee_bps / 10_000.0  # commission on option legs; spread = slippage
        self.underlying_cost_rate = (fee_bps + slippage_bps) / 10_000.0  # flat cost on hedge shares
        self.position = Position(cash=initial_capital)
        self.spot: float = 0.0
        self.t: pd.Timestamp | None = None
        self._bar_notional: float = 0.0
```

- [ ] **Step 5: Replace the pricing helpers to use the surface**

Replace `_price_leg` and `_leg_delta` (lines 60-77) with:

```python
    def _leg_iv(self, leg: OptionLeg) -> float:
        assert self.t is not None
        return self.surface.iv(self.spot, leg.strike, leg.expiry, self.t)

    def _price_leg(self, leg: OptionLeg) -> float:
        """Model MID price per contract-share (multiplier applied by callers)."""
        assert self.t is not None
        t_yrs = _years_between(self.t, leg.expiry)
        if t_yrs <= 0:
            return intrinsic_value(leg, self.spot)
        iv = self._leg_iv(leg)
        if leg.style == "american":
            return crr_price(leg.right, self.spot, leg.strike, t_yrs, iv, self.rate, style="american")
        return bs_price(leg.right, self.spot, leg.strike, t_yrs, iv, self.rate)

    def _leg_delta(self, leg: OptionLeg) -> float:
        # BSM delta for ALL legs, incl. American (a known phase-1 limitation).
        assert self.t is not None
        t_yrs = _years_between(self.t, leg.expiry)
        iv = self._leg_iv(leg)
        return bs_greeks(leg.right, self.spot, leg.strike, t_yrs, iv, self.rate).delta

    def _fill_price(self, mid: float, leg: OptionLeg, *, side: float) -> float:
        """Cross the spread: ``side > 0`` buys at the ask, ``side < 0`` sells at the bid."""
        assert self.t is not None
        m = math.log(leg.strike / self.spot) if self.spot > 0 else 0.0
        dte = max((leg.expiry - self.t).days, 0)
        half = self.spread.half_spread_frac(mid, m, dte)
        tick = self.spread.min_tick
        if side > 0:
            return max(mid * (1.0 + half), mid + tick)
        return max(min(mid * (1.0 - half), mid - tick), 0.0)
```

- [ ] **Step 6: Replace `add_leg` and `close_all_options` to fill at the spread**

Replace `add_leg` (lines 95-102) with:

```python
    def add_leg(self, leg: OptionLeg) -> None:
        """Open (buy/sell) an option leg, crossing the spread at its current price."""
        mid = self._price_leg(leg)
        side = 1.0 if leg.quantity > 0 else -1.0  # buy long pays ask; sell short hits bid
        fill = self._fill_price(mid, leg, side=side)
        notional = abs(leg.quantity) * fill * CONTRACT_MULTIPLIER
        self.position.cash -= leg.quantity * fill * CONTRACT_MULTIPLIER  # buy lowers cash
        self.position.cash -= notional * self.option_fee_rate
        self._bar_notional += notional
        self.position.legs.append(leg)
```

Replace `close_all_options` (lines 104-112) with:

```python
    def close_all_options(self) -> None:
        """Close every option leg, crossing the spread (sell longs at bid, buy back shorts at ask)."""
        for leg in self.position.legs:
            mid = self._price_leg(leg)
            side = -1.0 if leg.quantity > 0 else 1.0  # sell long at bid; buy back short at ask
            fill = self._fill_price(mid, leg, side=side)
            notional = abs(leg.quantity) * fill * CONTRACT_MULTIPLIER
            self.position.cash += leg.quantity * fill * CONTRACT_MULTIPLIER  # sell raises cash
            self.position.cash -= notional * self.option_fee_rate
            self._bar_notional += notional
        self.position.legs = []
```

- [ ] **Step 7: Update `hedge_to_delta` to use the renamed underlying cost rate**

In `hedge_to_delta` (line 121), change `self.position.cash -= notional * self.cost_rate` to:

```python
        self.position.cash -= notional * self.underlying_cost_rate
```

- [ ] **Step 8: Replace the `run_options_backtest` signature + add migration**

Replace the signature and the opening `if len(prices) < 2` guard (lines 140-156) with:

```python
def run_options_backtest(
    prices: pd.Series,
    strategy: OptionsStrategy,
    *,
    surface: VolSurface | None = None,
    spread: SpreadModel | None = None,
    vol: float | None = None,
    rate: float = 0.04,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
    n_trials: int = 1,
) -> BacktestResult:
    """Run an options strategy over a price path and return a BacktestResult.

    Provide a ``surface`` (e.g. ``realistic_surface(prices)`` or
    ``FlatSurface(vol)``) and optionally a ``spread`` model (defaults to
    frictionless ``NoSpread``). The legacy ``vol=`` argument is a deprecated alias
    for ``surface=FlatSurface(vol)``.
    """
    if vol is not None:
        if surface is not None:
            raise ValueError("pass either vol= or surface=, not both")
        warnings.warn(
            "vol= is deprecated; pass surface=FlatSurface(vol) or realistic_surface(prices)",
            DeprecationWarning,
            stacklevel=2,
        )
        surface = FlatSurface(vol)
    if surface is None:
        raise ValueError(
            "run_options_backtest requires surface= (e.g. realistic_surface(prices) "
            "or FlatSurface(vol))"
        )
    if spread is None:
        spread = NoSpread()
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to compute a return")

    engine = OptionsEngine(surface, spread, rate, fee_bps, slippage_bps, initial_capital)
```

- [ ] **Step 9: Update the result config block**

In the `config={...}` dict at the end of `run_options_backtest` (the `"vol": vol,` and `"rate": rate,` lines, ~190-194), replace `"vol": vol,` with:

```python
            "surface": type(surface).__name__,
            "spread": type(spread).__name__,
```

(Leave `"rate": rate,` and the other keys as they are.)

- [ ] **Step 10: Run to verify the new tests pass**

Run: `uv run pytest tests/test_options_engine_frictions.py -v`
Expected: PASS (5 tests).

- [ ] **Step 11: Commit**

```bash
git add tradinglib/backtest/options_engine.py tests/test_options_engine_frictions.py
git commit -m "feat(options): engine prices & fills via vol surface + bid/ask spread"
```

---

## Task 4: Migrate existing callers to the new API

**Files:**
- Modify: `tradinglib/options/simulate.py:100-108`
- Modify: `models/options/01-delta-hedged-long-option-spy/backtest.py:25-29, 93-100`
- Modify: `tests/test_options_engine.py` (existing `vol=` call sites)

- [ ] **Step 1: Migrate `simulate.py`**

Add to its imports (after `from tradinglib.backtest.options_engine import ...`, line 21):

```python
from tradinglib.options.surface import FlatSurface
```

In `run_simulation`, change the `run_options_backtest(...)` call (line 100-108) argument `vol=vol,` to:

```python
            surface=FlatSurface(vol),
```

- [ ] **Step 2: Migrate the seed model** (`models/options/01-delta-hedged-long-option-spy/backtest.py`)

Add to the imports (alongside the other `tradinglib.options` imports, ~line 27-29):

```python
from tradinglib.options.surface import FlatSurface
```

In `run_for_gui`, change the `run_options_backtest(...)` argument `vol=implied_vol,` (line 96) to:

```python
        surface=FlatSurface(implied_vol),
```

- [ ] **Step 3: Migrate the existing engine tests** (`tests/test_options_engine.py`)

Add to its imports (after line 11, `from tradinglib.options.instruments import OptionLeg`):

```python
from tradinglib.options.surface import FlatSurface
```

Replace every `vol=0.2` argument in `run_options_backtest(...)` calls with `surface=FlatSurface(0.2)`. There are eight call sites: lines 25, 37, 57, 76, 99, 136, 138, 150. Each currently reads `..., vol=0.2, ...` (or `vol=0.2,` on its own line) — change the `vol=0.2` token to `surface=FlatSurface(0.2)`. Leave all other arguments unchanged.

- [ ] **Step 4: Run the affected suites**

Run: `uv run pytest tests/test_options_engine.py tests/test_options_seed_model.py tests/test_options_simulate.py -v`
Expected: PASS, with **no** `DeprecationWarning` in the run summary.

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/simulate.py models/options/01-delta-hedged-long-option-spy/backtest.py tests/test_options_engine.py
git commit -m "refactor(options): migrate vol= callers to surface=FlatSurface(vol)"
```

---

## Task 5: Public API exports

**Files:**
- Modify: `tradinglib/options/__init__.py`
- Test: `tests/test_options_api.py`

- [ ] **Step 1: Write the failing test** (`tests/test_options_api.py`)

```python
"""The options package exposes the new surface/spread API at the top level."""
from __future__ import annotations

import tradinglib.options as o


def test_surface_and_spread_exported() -> None:
    for name in (
        "VolSurface", "FlatSurface", "ParametricSurface", "SurfaceParams",
        "realistic_surface", "realized_vol",
        "SpreadModel", "NoSpread", "ParametricSpread",
    ):
        assert hasattr(o, name), name
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_options_api.py -v`
Expected: FAIL (e.g. `FlatSurface` not an attribute of `tradinglib.options`).

- [ ] **Step 3: Implement** — append to `tradinglib/options/__init__.py` (after the existing module docstring + `from __future__ import annotations`):

```python
from tradinglib.options.spread import NoSpread, ParametricSpread, SpreadModel
from tradinglib.options.surface import (
    FlatSurface,
    ParametricSurface,
    SurfaceParams,
    VolSurface,
    realistic_surface,
    realized_vol,
)

__all__ = [
    "FlatSurface",
    "NoSpread",
    "ParametricSpread",
    "ParametricSurface",
    "SpreadModel",
    "SurfaceParams",
    "VolSurface",
    "realistic_surface",
    "realized_vol",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_options_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/__init__.py tests/test_options_api.py
git commit -m "feat(options): export surface + spread public API"
```

---

## Task 6: Directional demo — frictionless vs realistic

**Files:**
- Create: `models/options/02-directional-call-spy/backtest.py`
- Test: `tests/test_directional_call.py`

- [ ] **Step 1: Write the failing test** (`tests/test_directional_call.py`)

```python
"""The directional-call demo runs frictionless vs realistic and costs money under frictions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/options/02-directional-call-spy"


def _load_module():
    spec = importlib.util.spec_from_file_location("_directional_call", MODEL_DIR / "backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spread_reduces_equity() -> None:
    mod = _load_module()
    idx = pd.date_range("2023-01-01", periods=180, freq="B")
    prices = pd.Series(100.0 * (1.0002 ** np.arange(180)), index=idx)  # gentle uptrend
    out = mod.run_compare(prices, tenor_days=60, otm_pct=0.0)
    assert set(out) >= {"naive_flat", "surface_no_spread", "surface_with_spread"}
    # Holding the surface fixed, adding the spread can only cost money (same fills/marks).
    assert (
        out["surface_with_spread"].equity_curve.iloc[-1]
        <= out["surface_no_spread"].equity_curve.iloc[-1]
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_directional_call.py -v`
Expected: FAIL (model file does not exist).

- [ ] **Step 3: Implement** (`models/options/02-directional-call-spy/backtest.py`)

```python
"""Directional long call on SPY — frictionless vs realistic frictions.

Buys a ~2-month call (ATM or a configurable OTM offset), holds and rolls at
expiry. Runs the same strategy three ways — naive (flat vol, no spread), the
synthetic realized-vol surface without spread, and the surface plus a bid/ask
spread — so the headline is the P&L gap: the cost of paying real skew and spread
on a 2w-6mo option trade.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.options.instruments import OptionLeg
from tradinglib.options.spread import NoSpread, ParametricSpread
from tradinglib.options.surface import FlatSurface, realistic_surface

SYMBOL = "SPY"
START = "2023-01-01"
END = "2024-12-31"
TENOR_DAYS = 60
OTM_PCT = 0.0
FLAT_VOL = 0.18


class DirectionalCall:
    """Hold one long call; open at the configured OTM offset, roll at expiry."""

    def __init__(self, tenor_days: int = TENOR_DAYS, otm_pct: float = OTM_PCT) -> None:
        self.tenor_days = tenor_days
        self.otm_pct = otm_pct

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        if not engine.position.legs:
            strike = float(round(spot * (1.0 + self.otm_pct)))
            expiry = t + pd.Timedelta(days=self.tenor_days)
            engine.add_leg(OptionLeg("call", strike=strike, expiry=expiry, quantity=1.0))


def run_compare(prices: pd.Series, *, tenor_days: int = TENOR_DAYS, otm_pct: float = OTM_PCT) -> dict:
    """Run the strategy three ways and return the BacktestResults.

    - ``naive_flat``: the optimistic baseline (constant vol, no spread).
    - ``surface_no_spread``: realistic vol surface, but no spread.
    - ``surface_with_spread``: realistic surface + bid/ask spread.

    Comparing the last two isolates the spread cost (surface held fixed);
    comparing ``naive_flat`` against ``surface_with_spread`` is the full
    optimistic-vs-realistic headline.
    """
    surface = realistic_surface(prices)
    naive_flat = run_options_backtest(
        prices, DirectionalCall(tenor_days, otm_pct),
        surface=FlatSurface(FLAT_VOL), spread=NoSpread(),
    )
    surface_no_spread = run_options_backtest(
        prices, DirectionalCall(tenor_days, otm_pct),
        surface=surface, spread=NoSpread(),
    )
    surface_with_spread = run_options_backtest(
        prices, DirectionalCall(tenor_days, otm_pct),
        surface=surface, spread=ParametricSpread(),
    )
    return {
        "naive_flat": naive_flat,
        "surface_no_spread": surface_no_spread,
        "surface_with_spread": surface_with_spread,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_daily(SYMBOL, start=START, end=END)["close"]
    out = run_compare(prices)

    summary = {
        key: {
            "metrics": res.metrics,
            "final_equity": float(res.equity_curve.iloc[-1]),
        }
        for key, res in out.items()
    }
    (out_dir / "compare.json").write_text(json.dumps(summary, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(10, 5))
    out["naive_flat"].equity_curve.plot(ax=ax, label="Naive (flat vol, no spread)")
    out["surface_no_spread"].equity_curve.plot(ax=ax, label="Surface, no spread")
    out["surface_with_spread"].equity_curve.plot(ax=ax, label="Surface + spread (realistic)")
    ax.set_title(f"{SYMBOL} — directional call, naive vs realistic frictions")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "compare_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test, then the end-to-end run**

Run: `uv run pytest tests/test_directional_call.py -v`
Expected: PASS.
Run: `uv run python models/options/02-directional-call-spy/backtest.py`
Expected: prints a JSON summary; `models/options/02-directional-call-spy/results/compare.json` exists with `naive_flat`, `surface_no_spread`, and `surface_with_spread` blocks, and `surface_with_spread.final_equity <= surface_no_spread.final_equity`.

- [ ] **Step 5: Commit**

```bash
git add models/options/02-directional-call-spy/ tests/test_directional_call.py
git commit -m "feat(models): directional-call demo — frictionless vs realistic frictions"
```

---

## Task 7: Docs + full green gate

**Files:**
- Modify: `docs/methodology.md`

- [ ] **Step 1: Document the surface + spread** — append a section to `docs/methodology.md`:

```markdown
## Realistic options frictions — synthetic vol surface & spread

The options engine prices and fills through a `VolSurface` and a `SpreadModel`
(`tradinglib/options/surface.py`, `spread.py`) instead of a single constant vol.
`realistic_surface(prices)` anchors ATM implied vol to the underlying's trailing
realized vol (× a volatility-risk premium) and overlays parametric skew and term
structure; `ParametricSpread` fills option legs by crossing a bid/ask that widens
for out-of-the-money and short-dated contracts. The legacy `vol=` argument is a
deprecated alias for `surface=FlatSurface(vol)`.

This is a **stress / plausibility model, not a market-calibrated one**: it tests
whether an edge survives realistic-shaped vol regimes and frictions, not the exact
historical P&L of a specific contract (which needs real options-chain data). See
`models/options/02-directional-call-spy/backtest.py` for a frictionless-vs-realistic
comparison.
```

- [ ] **Step 2: Run the full suite + linters**

Run: `uv run pytest -q`
Expected: PASS (all tests, including pre-existing).
Run: `uv run ruff check tradinglib tests models && uv run ruff format --check tradinglib tests`
Expected: no errors.
Run: `uv run mypy tradinglib/options/surface.py tradinglib/options/spread.py tradinglib/backtest/options_engine.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add docs/methodology.md
git commit -m "docs: document synthetic options vol surface + spread frictions"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** VolSurface + FlatSurface + ParametricSurface + realized-vol calibration (Task 1) ✓ · SpreadModel + NoSpread + ParametricSpread (Task 2) ✓ · engine surface/spread fills + `vol=` deprecation/migration (Task 3) ✓ · caller migration keeping behavior bit-identical (Task 4) ✓ · public API (Task 5) ✓ · directional demo frictionless-vs-realistic (Task 6) ✓ · docs + honest-caveat (Task 7) ✓. Out-of-scope items (paid chain loader / SP3, surface-aware Monte Carlo, regime-dependent skew/Tier 3) are intentionally not implemented.
- **Naming consistency:** `VolSurface.iv(spot, strike, expiry, t)`, `FlatSurface(vol)`, `ParametricSurface(atm_vol, params)`, `SurfaceParams`, `realized_vol`, `realistic_surface`; `SpreadModel.half_spread_frac(mid, m, dte)` + `min_tick`, `NoSpread`, `ParametricSpread`; engine `surface`/`spread`/`option_fee_rate`/`underlying_cost_rate`/`_leg_iv`/`_fill_price`; `run_options_backtest(..., surface=, spread=, vol=)` — used identically across tasks and the `__init__` export list.
- **Backward-compat:** Task 4 keeps the seed model + existing engine tests bit-identical by routing through `FlatSurface(vol)` + `NoSpread` (the engine's default spread). The deprecated `vol=` path is covered by an explicit warns-and-matches test (Task 3).
- **Acceptance criteria mapping:** spec criteria 1-3 → Task 1 tests; 4 → Task 2 tests; 5 → Task 3 round-trip test; 6 → Task 3 deprecation test + Task 4 green run; 7 → Task 6 test; 8 → Task 7 gate.
```
