# Options Backtest & Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an options pricing/simulation core and a delta-hedged-long-option seed model (historical backtest + Monte Carlo outcome distribution) to the repo, wired into the Streamlit GUI.

**Architecture:** A new `tradinglib/options/` package holds pure pricing/Greeks/instrument/simulation primitives. A new `tradinglib/backtest/options_engine.py` marks a multi-leg `Position` to market over a price path, delta-hedges, rolls at expiry, and emits a standard `BacktestResult` via the existing `compute_metrics`. A new `models/options/` family carries the seed model and its GUI surface.

**Tech Stack:** Python 3.12, numpy, scipy.stats (already deps), pandas, pytest, Streamlit + plotly. No new runtime dependencies.

**Spec:** `docs/specs/2026-06-04-options-backtest-and-simulation-design.md`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `tradinglib/options/__init__.py` | Package exports |
| `tradinglib/options/pricing.py` | BSM price + Greeks (European); CRR tree (American); implied vol |
| `tradinglib/options/instruments.py` | `OptionLeg`, `Position`, intrinsic/expiry helpers |
| `tradinglib/options/simulate.py` | GBM Monte Carlo paths + `SimulationResult` + `run_simulation` |
| `tradinglib/backtest/options_engine.py` | `OptionsEngine`, `OptionsStrategy`, `run_options_backtest` |
| `models/options/01-delta-hedged-long-option-spy/backtest.py` | Seed model: strategy + `main()` + `run_for_gui()` |
| `models/options/01-delta-hedged-long-option-spy/model.md` | Frontmatter for `MODELS.md` / GUI discovery |
| `app/ui/options_view.py` | Payoff/Greeks plot + Monte Carlo histogram |
| `tests/test_options_pricing.py` | Pricing/Greeks/IV/CRR tests |
| `tests/test_options_instruments.py` | Instrument model tests |
| `tests/test_options_engine.py` | Engine mark-to-market / hedge / expiry tests |
| `tests/test_options_simulate.py` | GBM + simulation tests |

Conventions to follow (verified in the existing code):
- Tests use plain `pytest` with `from __future__ import annotations`, numpy/pandas fixtures (see `tests/test_backtest_engine.py`).
- Run tests with `uv run pytest <path> -v`.
- `BacktestResult` lives at `tradinglib.backtest.BacktestResult` with fields `equity_curve, returns, position, turnover, metrics, config`. `config` MUST include `initial_capital` (the GUI's `results_view` reads it for the buy-and-hold benchmark).
- Models are auto-discovered from `model.md` frontmatter; the GUI runs them via `run_for_gui(start, end, **params)` returning a dict with at least `data` (DataFrame with a `close` column), `result` (BacktestResult), `symbol`, `params`.

---

## Task 1: Scaffold the `tradinglib/options/` package

**Files:**
- Create: `tradinglib/options/__init__.py`
- Test: `tests/test_options_pricing.py` (smoke import only for now)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_pricing.py
"""Tests for tradinglib.options pricing primitives."""

from __future__ import annotations

import math

import pytest


def test_package_imports() -> None:
    import tradinglib.options  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_pricing.py::test_package_imports -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradinglib.options'`

- [ ] **Step 3: Create the package**

```python
# tradinglib/options/__init__.py
"""Options pricing, Greeks, instruments, and Monte Carlo simulation.

Pure, dependency-light primitives (numpy + scipy.stats) used by the options
backtest engine in ``tradinglib.backtest.options_engine`` and the seed model
under ``models/options/``.
"""

from __future__ import annotations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_options_pricing.py::test_package_imports -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/__init__.py tests/test_options_pricing.py
git commit -m "feat(options): scaffold options package"
```

---

## Task 2: Black-Scholes price (`bs_price`)

**Files:**
- Create: `tradinglib/options/pricing.py`
- Test: `tests/test_options_pricing.py`

Reference values (S=100, K=100, T=1, r=0.05, vol=0.20, no dividend): call ≈ 10.4506, put ≈ 5.5735 (put-call parity: `call - put = S - K·e^{-rT}`).

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_options_pricing.py
from tradinglib.options.pricing import bs_price


def test_bs_call_reference_value() -> None:
    price = bs_price("call", spot=100, strike=100, t=1.0, vol=0.20, rate=0.05)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_bs_put_reference_value() -> None:
    price = bs_price("put", spot=100, strike=100, t=1.0, vol=0.20, rate=0.05)
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity() -> None:
    call = bs_price("call", 100, 110, 0.5, 0.25, 0.03)
    put = bs_price("put", 100, 110, 0.5, 0.25, 0.03)
    spot, strike, rate, t = 100, 110, 0.03, 0.5
    assert call - put == pytest.approx(spot - strike * math.exp(-rate * t), abs=1e-9)


def test_bs_at_expiry_is_intrinsic() -> None:
    assert bs_price("call", 120, 100, 0.0, 0.2, 0.05) == pytest.approx(20.0)
    assert bs_price("put", 80, 100, 0.0, 0.2, 0.05) == pytest.approx(20.0)
    assert bs_price("call", 80, 100, 0.0, 0.2, 0.05) == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: FAIL with `ImportError: cannot import name 'bs_price'`

- [ ] **Step 3: Implement `bs_price`**

```python
# tradinglib/options/pricing.py
"""Option pricing and Greeks.

European options use the closed-form Black-Scholes-Merton model; American
options use a Cox-Ross-Rubinstein binomial tree that checks early exercise at
each node. All functions are pure and accept scalar floats (vectorize with
``numpy.vectorize`` at the call site if needed).

Conventions
-----------
- ``t`` is time to expiry in YEARS. ``t <= 0`` returns intrinsic value.
- ``vol`` and ``rate`` are annualized, expressed as decimals (0.20 = 20%).
- ``div`` is a continuous dividend yield (annualized decimal), default 0.
- ``right`` is ``"call"`` or ``"put"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

Right = Literal["call", "put"]


def _intrinsic(right: Right, spot: float, strike: float) -> float:
    if right == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _d1_d2(spot: float, strike: float, t: float, vol: float, rate: float, div: float) -> tuple[float, float]:
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return d1, d2


def bs_price(
    right: Right,
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    div: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European option."""
    if t <= 0 or vol <= 0:
        return _intrinsic(right, spot, strike)
    d1, d2 = _d1_d2(spot, strike, t, vol, rate, div)
    disc = math.exp(-rate * t)
    carry = math.exp(-div * t)
    if right == "call":
        return spot * carry * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    return strike * disc * norm.cdf(-d2) - spot * carry * norm.cdf(-d1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/pricing.py tests/test_options_pricing.py
git commit -m "feat(options): Black-Scholes European pricing"
```

---

## Task 3: Black-Scholes Greeks (`bs_greeks`)

**Files:**
- Modify: `tradinglib/options/pricing.py`
- Test: `tests/test_options_pricing.py`

Reference Greeks for the call (S=100, K=100, T=1, r=0.05, vol=0.20): delta ≈ 0.6368, gamma ≈ 0.018762, vega ≈ 0.3752 (per 1% vol → divide raw vega by 100). We return RAW vega (per 1.0 of vol, = 37.52) and verify delta/gamma against finite differences too.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_options_pricing.py
from tradinglib.options.pricing import bs_greeks


def test_call_delta_reference() -> None:
    g = bs_greeks("call", 100, 100, 1.0, 0.20, 0.05)
    assert g.delta == pytest.approx(0.6368, abs=1e-3)


def test_put_delta_is_call_delta_minus_one() -> None:
    c = bs_greeks("call", 100, 100, 1.0, 0.20, 0.05)
    p = bs_greeks("put", 100, 100, 1.0, 0.20, 0.05)
    assert p.delta == pytest.approx(c.delta - 1.0, abs=1e-9)


def test_delta_matches_finite_difference() -> None:
    eps = 1e-4
    up = bs_price("call", 100 + eps, 100, 1.0, 0.20, 0.05)
    dn = bs_price("call", 100 - eps, 100, 1.0, 0.20, 0.05)
    fd_delta = (up - dn) / (2 * eps)
    assert bs_greeks("call", 100, 100, 1.0, 0.20, 0.05).delta == pytest.approx(fd_delta, abs=1e-4)


def test_gamma_reference() -> None:
    g = bs_greeks("call", 100, 100, 1.0, 0.20, 0.05)
    assert g.gamma == pytest.approx(0.018762, abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: FAIL with `ImportError: cannot import name 'bs_greeks'`

- [ ] **Step 3: Implement `Greeks` + `bs_greeks`**

Add to `tradinglib/options/pricing.py`:

```python
@dataclass(frozen=True)
class Greeks:
    """First-order Greeks. ``vega``/``rho`` are per 1.0 of vol/rate (per 100%).

    Divide by 100 for per-1%-move conventions. ``theta`` is per YEAR; divide by
    365 for per-calendar-day.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bs_greeks(
    right: Right,
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    div: float = 0.0,
) -> Greeks:
    """First-order Black-Scholes Greeks for a European option."""
    if t <= 0 or vol <= 0:
        # At/after expiry: delta is a step function, other Greeks vanish.
        if right == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(spot, strike, t, vol, rate, div)
    pdf = norm.pdf(d1)
    disc = math.exp(-rate * t)
    carry = math.exp(-div * t)
    sqrt_t = math.sqrt(t)

    gamma = carry * pdf / (spot * vol * sqrt_t)
    vega = spot * carry * pdf * sqrt_t
    if right == "call":
        delta = carry * norm.cdf(d1)
        theta = (
            -spot * carry * pdf * vol / (2 * sqrt_t)
            - rate * strike * disc * norm.cdf(d2)
            + div * spot * carry * norm.cdf(d1)
        )
        rho = strike * t * disc * norm.cdf(d2)
    else:
        delta = -carry * norm.cdf(-d1)
        theta = (
            -spot * carry * pdf * vol / (2 * sqrt_t)
            + rate * strike * disc * norm.cdf(-d2)
            - div * spot * carry * norm.cdf(-d1)
        )
        rho = -strike * t * disc * norm.cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/pricing.py tests/test_options_pricing.py
git commit -m "feat(options): Black-Scholes Greeks"
```

---

## Task 4: Implied volatility (`implied_vol`)

**Files:**
- Modify: `tradinglib/options/pricing.py`
- Test: `tests/test_options_pricing.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_options_pricing.py
from tradinglib.options.pricing import implied_vol


def test_implied_vol_round_trips() -> None:
    true_vol = 0.27
    price = bs_price("call", 100, 105, 0.5, true_vol, 0.04)
    assert implied_vol(price, "call", 100, 105, 0.5, 0.04) == pytest.approx(true_vol, abs=1e-4)


def test_implied_vol_put_round_trips() -> None:
    true_vol = 0.18
    price = bs_price("put", 100, 95, 0.75, true_vol, 0.04)
    assert implied_vol(price, "put", 100, 95, 0.75, 0.04) == pytest.approx(true_vol, abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: FAIL with `ImportError: cannot import name 'implied_vol'`

- [ ] **Step 3: Implement `implied_vol`**

Add to `tradinglib/options/pricing.py` (uses `brentq` from scipy — add `from scipy.optimize import brentq` to the imports):

```python
def implied_vol(
    price: float,
    right: Right,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div: float = 0.0,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Invert Black-Scholes for the volatility implied by an observed price.

    Uses Brent's method on ``[lo, hi]``. Raises ``ValueError`` if the price is
    below intrinsic (no real implied vol exists).
    """
    if price < _intrinsic(right, spot, strike) - 1e-9:
        raise ValueError(f"price {price} is below intrinsic value")

    def objective(vol: float) -> float:
        return bs_price(right, spot, strike, t, vol, rate, div) - price

    return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/pricing.py tests/test_options_pricing.py
git commit -m "feat(options): implied volatility inversion"
```

---

## Task 5: CRR binomial tree (`crr_price`)

**Files:**
- Modify: `tradinglib/options/pricing.py`
- Test: `tests/test_options_pricing.py`

Properties to verify: a European CRR price converges to BS as steps↑; an American call with no dividends equals the European call (never optimal to exercise early); an American put is ≥ the European put.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_options_pricing.py
from tradinglib.options.pricing import crr_price


def test_crr_european_converges_to_bs() -> None:
    bs = bs_price("call", 100, 100, 1.0, 0.20, 0.05)
    crr = crr_price("call", 100, 100, 1.0, 0.20, 0.05, style="european", steps=2000)
    assert crr == pytest.approx(bs, abs=1e-2)


def test_american_call_equals_european_without_dividends() -> None:
    euro = crr_price("call", 100, 100, 1.0, 0.20, 0.05, style="european", steps=1000)
    amer = crr_price("call", 100, 100, 1.0, 0.20, 0.05, style="american", steps=1000)
    assert amer == pytest.approx(euro, abs=1e-6)


def test_american_put_at_least_european_put() -> None:
    euro = crr_price("put", 100, 100, 1.0, 0.20, 0.05, style="european", steps=1000)
    amer = crr_price("put", 100, 100, 1.0, 0.20, 0.05, style="american", steps=1000)
    assert amer >= euro - 1e-9
    assert amer > euro  # early exercise has positive value for an ATM put here
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: FAIL with `ImportError: cannot import name 'crr_price'`

- [ ] **Step 3: Implement `crr_price`** (uses numpy — add `import numpy as np` to imports)

```python
def crr_price(
    right: Right,
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    style: Literal["european", "american"] = "american",
    div: float = 0.0,
    steps: int = 512,
) -> float:
    """Cox-Ross-Rubinstein binomial price. American style checks early exercise."""
    if t <= 0 or vol <= 0:
        return _intrinsic(right, spot, strike)

    dt = t / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rate * dt)
    p = (math.exp((rate - div) * dt) - d) / (u - d)

    # Terminal spot prices: spot * u^j * d^(steps-j) for j in 0..steps.
    j = np.arange(steps + 1)
    spot_t = spot * u**j * d ** (steps - j)
    if right == "call":
        values = np.maximum(spot_t - strike, 0.0)
    else:
        values = np.maximum(strike - spot_t, 0.0)

    # Backward induction.
    for step in range(steps, 0, -1):
        values = disc * (p * values[1:step + 1] + (1 - p) * values[0:step])
        if style == "american":
            j = np.arange(step)
            spot_nodes = spot * u**j * d ** (step - 1 - j)
            if right == "call":
                exercise = np.maximum(spot_nodes - strike, 0.0)
            else:
                exercise = np.maximum(strike - spot_nodes, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_pricing.py -v`
Expected: PASS (all pricing tests)

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/pricing.py tests/test_options_pricing.py
git commit -m "feat(options): CRR binomial pricer with American early exercise"
```

---

## Task 6: Instrument model (`OptionLeg`, `Position`)

**Files:**
- Create: `tradinglib/options/instruments.py`
- Test: `tests/test_options_instruments.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_options_instruments.py
"""Tests for the options instrument model."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.options.instruments import (
    CONTRACT_MULTIPLIER,
    OptionLeg,
    Position,
    intrinsic_value,
)


def _leg(**kw) -> OptionLeg:
    base = dict(right="call", strike=100.0, expiry=pd.Timestamp("2025-01-31"), quantity=1.0)
    base.update(kw)
    return OptionLeg(**base)


def test_intrinsic_value_call_and_put() -> None:
    assert intrinsic_value(_leg(right="call", strike=100), spot=120) == 20.0
    assert intrinsic_value(_leg(right="call", strike=100), spot=80) == 0.0
    assert intrinsic_value(_leg(right="put", strike=100), spot=80) == 20.0
    assert intrinsic_value(_leg(right="put", strike=100), spot=120) == 0.0


def test_option_leg_is_frozen() -> None:
    leg = _leg()
    with pytest.raises(Exception):
        leg.strike = 105.0  # type: ignore[misc]


def test_position_intrinsic_value_sums_legs_shares_cash() -> None:
    pos = Position(
        legs=[_leg(right="call", strike=100, quantity=2.0)],
        shares=10.0,
        cash=500.0,
    )
    # 2 contracts * intrinsic(20) * 100 + 10 shares * 120 + 500 cash
    expected = 2.0 * 20.0 * CONTRACT_MULTIPLIER + 10.0 * 120.0 + 500.0
    assert pos.intrinsic_value(spot=120.0) == pytest.approx(expected)


def test_short_leg_has_negative_quantity() -> None:
    pos = Position(legs=[_leg(right="put", strike=100, quantity=-1.0)])
    # short a put, spot 80 → liability of intrinsic(20) * 100
    assert pos.intrinsic_value(spot=80.0) == pytest.approx(-20.0 * CONTRACT_MULTIPLIER)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_instruments.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradinglib.options.instruments'`

- [ ] **Step 3: Implement `instruments.py`**

```python
# tradinglib/options/instruments.py
"""Multi-leg options position model.

An :class:`OptionLeg` is one contract line (right, strike, expiry, signed
quantity). A :class:`Position` is the unit the backtest engine marks to market:
a list of legs plus underlying ``shares`` and ``cash``. The standard equity
contract multiplier is 100 (one contract controls 100 shares).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

CONTRACT_MULTIPLIER = 100.0

Right = Literal["call", "put"]
Style = Literal["european", "american"]


@dataclass(frozen=True)
class OptionLeg:
    """One option contract line. ``quantity`` is signed: +long, -short."""

    right: Right
    strike: float
    expiry: pd.Timestamp
    quantity: float
    style: Style = "european"
    underlying: str = "SPY"


def intrinsic_value(leg: OptionLeg, spot: float) -> float:
    """Per-contract intrinsic value of one leg's option (unsigned by quantity)."""
    if leg.right == "call":
        return max(spot - leg.strike, 0.0)
    return max(leg.strike - spot, 0.0)


@dataclass
class Position:
    """A multi-leg options position plus underlying shares and cash."""

    legs: list[OptionLeg] = field(default_factory=list)
    shares: float = 0.0
    cash: float = 0.0

    def intrinsic_value(self, spot: float) -> float:
        """Mark every leg at intrinsic value (used at/after expiry)."""
        options = sum(
            leg.quantity * intrinsic_value(leg, spot) * CONTRACT_MULTIPLIER for leg in self.legs
        )
        return options + self.shares * spot + self.cash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_instruments.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/instruments.py tests/test_options_instruments.py
git commit -m "feat(options): multi-leg position model"
```

---

## Task 7: Options backtest engine (`options_engine.py`)

**Files:**
- Create: `tradinglib/backtest/options_engine.py`
- Test: `tests/test_options_engine.py`

The engine processes a price path bar-by-bar. Per bar it: (1) settles any expired legs to intrinsic cash, (2) resets the bar's traded-notional counter, (3) calls `strategy.on_bar`, (4) records equity (`cash + shares·spot + option mark-to-market`), net share-delta, and turnover. Trades flow through helper methods that update cash and accrue costs. Establishing a position at a bar's spot is cash-neutral except costs, so P&L accrues from the next bar's spot move — same lag-1 no-look-ahead behavior as the linear engine.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_options_engine.py
"""Tests for the options backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest import BacktestResult
from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.options.instruments import CONTRACT_MULTIPLIER, OptionLeg


@pytest.fixture
def flat_path() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    return pd.Series(100.0, index=idx)


def test_returns_backtest_result(flat_path: pd.Series) -> None:
    class DoNothing:
        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            return None

    result = run_options_backtest(flat_path, DoNothing(), vol=0.2, rate=0.04)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(flat_path)
    assert "initial_capital" in result.config


def test_do_nothing_keeps_equity_flat(flat_path: pd.Series) -> None:
    class DoNothing:
        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            return None

    result = run_options_backtest(
        flat_path, DoNothing(), vol=0.2, rate=0.04, initial_capital=100_000.0
    )
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_long_call_gains_when_spot_rises() -> None:
    # Spot ramps from 100 to 120; buy one ATM call at bar 0, hold to near expiry.
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.Series(np.linspace(100.0, 120.0, 30), index=idx)
    expiry = idx[-1]

    class BuyAndHoldCall:
        def __init__(self) -> None:
            self.opened = False

        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            if not self.opened:
                engine.add_leg(OptionLeg("call", strike=100.0, expiry=expiry, quantity=1.0))
                self.opened = True

    result = run_options_backtest(
        prices, BuyAndHoldCall(), vol=0.2, rate=0.04, fee_bps=0, slippage_bps=0
    )
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_expired_leg_settles_to_intrinsic() -> None:
    # Buy an ITM call that expires mid-path; afterwards equity holds the settled cash.
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    prices = pd.Series(110.0, index=idx)
    expiry = idx[10]

    class BuyOnce:
        def __init__(self) -> None:
            self.opened = False

        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            if not self.opened:
                engine.add_leg(OptionLeg("call", strike=100.0, expiry=expiry, quantity=1.0))
                self.opened = True

    result = run_options_backtest(
        prices, BuyOnce(), vol=0.2, rate=0.04, fee_bps=0, slippage_bps=0
    )
    # No legs remain at the end; equity = cash only, flat across the tail.
    assert result.equity_curve.iloc[-1] == pytest.approx(result.equity_curve.iloc[-2], abs=1e-6)


def test_delta_hedged_position_is_insensitive_to_small_moves() -> None:
    # A delta-hedged long call should barely move when spot ticks slightly;
    # compare to an unhedged long call over the same one-bar move.
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.Series([100.0, 100.5, 100.5], index=idx)
    expiry = idx[-1] + pd.Timedelta(days=30)

    class Hedged:
        def __init__(self) -> None:
            self.opened = False

        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            if not self.opened:
                engine.add_leg(OptionLeg("call", strike=100.0, expiry=expiry, quantity=1.0))
                self.opened = True
            engine.hedge_to_delta(0.0)

    class Unhedged:
        def __init__(self) -> None:
            self.opened = False

        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            if not self.opened:
                engine.add_leg(OptionLeg("call", strike=100.0, expiry=expiry, quantity=1.0))
                self.opened = True

    hedged = run_options_backtest(prices, Hedged(), vol=0.2, rate=0.04, fee_bps=0, slippage_bps=0)
    unhedged = run_options_backtest(prices, Unhedged(), vol=0.2, rate=0.04, fee_bps=0, slippage_bps=0)
    hedged_move = abs(hedged.equity_curve.iloc[1] - hedged.equity_curve.iloc[0])
    unhedged_move = abs(unhedged.equity_curve.iloc[1] - unhedged.equity_curve.iloc[0])
    assert hedged_move < unhedged_move
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradinglib.backtest.options_engine'`

- [ ] **Step 3: Implement `options_engine.py`**

```python
# tradinglib/backtest/options_engine.py
"""Event-driven options backtest engine.

Marks a multi-leg :class:`~tradinglib.options.instruments.Position` to market
over a price path (real or simulated). Unlike the vectorized engine, options
P&L is the *change in option value*, which is nonlinear in the underlying — so
this engine computes its own equity curve by mark-to-market rather than the
``position * return`` math in :func:`tradinglib.backtest.run_backtest`. It then
calls the shared :func:`compute_metrics` so results stay comparable.

``BacktestResult`` fields are reinterpreted for options (see
``docs/methodology.md``): ``position`` is net portfolio delta as a fraction of
equity; ``turnover`` is traded notional / equity.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from tradinglib.backtest.engine import BacktestResult
from tradinglib.backtest.metrics import compute_metrics
from tradinglib.options.instruments import CONTRACT_MULTIPLIER, OptionLeg, Position, intrinsic_value
from tradinglib.options.pricing import bs_greeks, bs_price, crr_price


class OptionsStrategy(Protocol):
    """Strategies implement a single per-bar callback."""

    def on_bar(self, engine: "OptionsEngine", t: pd.Timestamp, spot: float) -> None: ...


def _years_between(now: pd.Timestamp, expiry: pd.Timestamp) -> float:
    return max((expiry - now).days, 0) / 365.0


class OptionsEngine:
    """Per-bar dispatcher. The strategy adjusts the position via these methods;
    the engine handles pricing, cash accounting, costs, and turnover."""

    def __init__(self, vol: float, rate: float, fee_bps: float, slippage_bps: float, initial_capital: float) -> None:
        self.vol = vol
        self.rate = rate
        self.cost_rate = (fee_bps + slippage_bps) / 10_000.0
        self.position = Position(cash=initial_capital)
        self.spot: float = 0.0
        self.t: pd.Timestamp | None = None
        self._bar_notional: float = 0.0

    # --- pricing helpers ----------------------------------------------------
    def _price_leg(self, leg: OptionLeg) -> float:
        t_yrs = _years_between(self.t, leg.expiry)
        if t_yrs <= 0:
            return intrinsic_value(leg, self.spot)
        if leg.style == "american":
            return crr_price(leg.right, self.spot, leg.strike, t_yrs, self.vol, self.rate, style="american")
        return bs_price(leg.right, self.spot, leg.strike, t_yrs, self.vol, self.rate)

    def _leg_delta(self, leg: OptionLeg) -> float:
        t_yrs = _years_between(self.t, leg.expiry)
        return bs_greeks(leg.right, self.spot, leg.strike, t_yrs, self.vol, self.rate).delta

    def option_value(self) -> float:
        return sum(self._price_leg(leg) * leg.quantity * CONTRACT_MULTIPLIER for leg in self.position.legs)

    def net_delta_shares(self) -> float:
        """Net delta expressed in shares of the underlying (incl. hedge shares)."""
        opt = sum(self._leg_delta(leg) * leg.quantity * CONTRACT_MULTIPLIER for leg in self.position.legs)
        return opt + self.position.shares

    def equity(self) -> float:
        return self.position.cash + self.position.shares * self.spot + self.option_value()

    # --- trading API (called by strategies) ---------------------------------
    def add_leg(self, leg: OptionLeg) -> None:
        """Open (buy/sell) an option leg at its current model price."""
        price = self._price_leg(leg)
        notional = abs(leg.quantity) * price * CONTRACT_MULTIPLIER
        self.position.cash -= leg.quantity * price * CONTRACT_MULTIPLIER  # buy lowers cash
        self.position.cash -= notional * self.cost_rate
        self._bar_notional += notional
        self.position.legs.append(leg)

    def close_all_options(self) -> None:
        """Close every option leg at current model price."""
        for leg in self.position.legs:
            price = self._price_leg(leg)
            notional = abs(leg.quantity) * price * CONTRACT_MULTIPLIER
            self.position.cash += leg.quantity * price * CONTRACT_MULTIPLIER  # sell raises cash
            self.position.cash -= notional * self.cost_rate
            self._bar_notional += notional
        self.position.legs = []

    def hedge_to_delta(self, target_share_delta: float = 0.0) -> None:
        """Trade underlying shares so net share-delta equals the target."""
        delta_shares = target_share_delta - self.net_delta_shares()
        if delta_shares == 0.0:
            return
        notional = abs(delta_shares) * self.spot
        self.position.cash -= delta_shares * self.spot
        self.position.cash -= notional * self.cost_rate
        self._bar_notional += notional
        self.position.shares += delta_shares

    # --- internal -----------------------------------------------------------
    def _settle_expiries(self) -> None:
        survivors: list[OptionLeg] = []
        for leg in self.position.legs:
            if (leg.expiry - self.t).days <= 0:
                self.position.cash += leg.quantity * intrinsic_value(leg, self.spot) * CONTRACT_MULTIPLIER
            else:
                survivors.append(leg)
        self.position.legs = survivors


def run_options_backtest(
    prices: pd.Series,
    strategy: OptionsStrategy,
    *,
    vol: float,
    rate: float = 0.04,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run an options strategy over a price path and return a BacktestResult."""
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to compute a return")

    engine = OptionsEngine(vol, rate, fee_bps, slippage_bps, initial_capital)
    equities: list[float] = []
    deltas: list[float] = []
    turnovers: list[float] = []

    for t, spot in prices.items():
        engine.t = pd.Timestamp(t)
        engine.spot = float(spot)
        engine._settle_expiries()
        engine._bar_notional = 0.0
        strategy.on_bar(engine, engine.t, engine.spot)

        eq = engine.equity()
        equities.append(eq)
        deltas.append(engine.net_delta_shares() * engine.spot / eq if eq != 0 else 0.0)
        turnovers.append(engine._bar_notional / eq if eq != 0 else 0.0)

    equity_curve = pd.Series(equities, index=prices.index, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)
    position = pd.Series(deltas, index=prices.index, name="delta_fraction")
    turnover = pd.Series(turnovers, index=prices.index, name="turnover")
    metrics = compute_metrics(returns, equity_curve, periods_per_year=periods_per_year)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        position=position,
        turnover=turnover,
        metrics=metrics,
        config={
            "initial_capital": initial_capital,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "periods_per_year": periods_per_year,
            "vol": vol,
            "rate": rate,
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_engine.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tradinglib/backtest/options_engine.py tests/test_options_engine.py
git commit -m "feat(options): mark-to-market options backtest engine"
```

---

## Task 8: GBM Monte Carlo paths (`gbm_paths`)

**Files:**
- Create: `tradinglib/options/simulate.py`
- Test: `tests/test_options_simulate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_options_simulate.py
"""Tests for GBM path simulation and the strategy outcome simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tradinglib.options.simulate import gbm_paths


def test_paths_shape_and_start() -> None:
    paths = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=252, n_paths=1000, seed=0)
    assert paths.shape == (1000, 253)  # days + 1 (includes the t=0 column)
    assert np.allclose(paths[:, 0], 100.0)


def test_paths_are_float32() -> None:
    paths = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=10, n_paths=10, seed=0)
    assert paths.dtype == np.float32


def test_terminal_mean_matches_risk_neutral_drift() -> None:
    spot, vol, rate, days, n = 100.0, 0.2, 0.05, 252, 200_000
    paths = gbm_paths(spot=spot, vol=vol, rate=rate, days=days, n_paths=n, seed=42)
    terminal_mean = float(paths[:, -1].mean())
    t_years = days / 252
    expected = spot * math.exp(rate * t_years)
    assert terminal_mean == pytest.approx(expected, rel=0.01)


def test_seed_is_deterministic() -> None:
    a = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=20, n_paths=50, seed=7)
    b = gbm_paths(spot=100.0, vol=0.2, rate=0.05, days=20, n_paths=50, seed=7)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradinglib.options.simulate'`

- [ ] **Step 3: Implement `gbm_paths`**

```python
# tradinglib/options/simulate.py
"""Monte Carlo simulation for options strategies.

:func:`gbm_paths` generates geometric-Brownian-motion underlying paths under
the risk-neutral measure. :func:`run_simulation` runs a strategy across many
paths and aggregates to a :class:`SimulationResult` distribution.

Memory: paths are ``float32`` and the simulation aggregates to per-path P&L
without retaining per-leg histories, keeping peak memory well under the ~1 GB
Streamlit Community Cloud cap. Callers should still bound ``n_paths``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def gbm_paths(
    spot: float,
    vol: float,
    rate: float,
    days: int,
    n_paths: int,
    *,
    steps_per_day: int = 1,
    seed: int | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """Simulate GBM underlying paths. Returns shape ``(n_paths, days*steps_per_day + 1)``.

    Column 0 is the starting spot. ``dt`` is in years (``1 / (252*steps_per_day)``),
    so the terminal horizon is ``days / 252`` years.
    """
    rng = np.random.default_rng(seed)
    n_steps = days * steps_per_day
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * steps_per_day)
    drift = (rate - 0.5 * vol * vol) * dt
    diffusion = vol * math.sqrt(dt)

    z = rng.standard_normal((n_paths, n_steps)).astype(dtype)
    log_increments = drift + diffusion * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = (spot * np.exp(log_paths)).astype(dtype)
    start = np.full((n_paths, 1), spot, dtype=dtype)
    return np.concatenate([start, paths], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_simulate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/simulate.py tests/test_options_simulate.py
git commit -m "feat(options): GBM Monte Carlo path generator"
```

---

## Task 9: Strategy outcome simulation (`SimulationResult`, `run_simulation`)

**Files:**
- Modify: `tradinglib/options/simulate.py`
- Test: `tests/test_options_simulate.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_options_simulate.py
import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.options.instruments import OptionLeg
from tradinglib.options.simulate import SimulationResult, run_simulation


def _long_call_factory(expiry_index: int):
    """Return a strategy-factory that buys one ATM call at bar 0."""

    def factory():
        class BuyCall:
            def __init__(self) -> None:
                self.opened = False
                self.expiry_index = expiry_index

            def on_bar(self, engine: OptionsEngine, t, spot) -> None:
                if not self.opened:
                    expiry = t + pd.Timedelta(days=30)
                    engine.add_leg(OptionLeg("call", strike=round(spot), expiry=expiry, quantity=1.0))
                    self.opened = True

        return BuyCall()

    return factory


def test_run_simulation_returns_distribution() -> None:
    result = run_simulation(
        _long_call_factory(20),
        spot=100.0,
        vol=0.2,
        rate=0.04,
        days=20,
        n_paths=500,
        seed=1,
    )
    assert isinstance(result, SimulationResult)
    assert result.pnl_distribution.shape == (500,)
    assert 0.0 <= result.prob_of_profit <= 1.0
    assert set(result.percentiles) == {5, 25, 50, 75, 95}
    assert result.percentiles[5] <= result.percentiles[95]


def test_simulation_respects_max_paths_cap() -> None:
    result = run_simulation(
        _long_call_factory(10),
        spot=100.0,
        vol=0.2,
        rate=0.04,
        days=10,
        n_paths=10_000,
        max_paths=1_000,
        seed=2,
    )
    assert result.pnl_distribution.shape == (1_000,)
    assert result.truncated is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_simulate.py -v`
Expected: FAIL with `ImportError: cannot import name 'SimulationResult'`

- [ ] **Step 3: Implement `SimulationResult` + `run_simulation`**

Add to `tradinglib/options/simulate.py` (add `from collections.abc import Callable` and `import pandas as pd` to imports):

```python
@dataclass
class SimulationResult:
    """Distribution of strategy P&L across simulated paths."""

    pnl_distribution: np.ndarray          # terminal P&L per path (initial capital subtracted)
    percentiles: dict[int, float]         # {5, 25, 50, 75, 95} -> P&L
    prob_of_profit: float
    expected_shortfall: float             # mean P&L of the worst 5% of paths
    mean: float
    std: float
    sample_paths: np.ndarray              # a handful of underlying paths for plotting
    truncated: bool                       # True if n_paths was clamped to max_paths


def run_simulation(
    strategy_factory: "Callable[[], object]",
    *,
    spot: float,
    vol: float,
    rate: float,
    days: int,
    n_paths: int,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    seed: int | None = None,
    max_paths: int = 20_000,
    n_sample_paths: int = 40,
) -> SimulationResult:
    """Run ``strategy_factory()`` (a fresh strategy per path) across GBM paths.

    Returns a :class:`SimulationResult`. ``n_paths`` is clamped to ``max_paths``
    to bound memory/runtime; the returned ``truncated`` flag records whether the
    clamp fired so callers can surface it.
    """
    truncated = n_paths > max_paths
    n_paths = min(n_paths, max_paths)

    paths = gbm_paths(spot, vol, rate, days, n_paths, seed=seed)
    index = pd.bdate_range("2024-01-01", periods=paths.shape[1])

    pnl = np.empty(n_paths, dtype=np.float64)
    for i in range(n_paths):
        price_series = pd.Series(paths[i].astype(np.float64), index=index)
        result = run_options_backtest(
            price_series,
            strategy_factory(),
            vol=vol,
            rate=rate,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        pnl[i] = float(result.equity_curve.iloc[-1]) - initial_capital

    pct_levels = [5, 25, 50, 75, 95]
    pct_values = np.percentile(pnl, pct_levels)
    percentiles = {lvl: float(v) for lvl, v in zip(pct_levels, pct_values)}
    worst_5pct = pnl[pnl <= percentiles[5]]
    expected_shortfall = float(worst_5pct.mean()) if worst_5pct.size else float(pnl.min())

    return SimulationResult(
        pnl_distribution=pnl,
        percentiles=percentiles,
        prob_of_profit=float((pnl > 0).mean()),
        expected_shortfall=expected_shortfall,
        mean=float(pnl.mean()),
        std=float(pnl.std(ddof=0)),
        sample_paths=paths[:n_sample_paths],
        truncated=truncated,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_simulate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradinglib/options/simulate.py tests/test_options_simulate.py
git commit -m "feat(options): strategy outcome Monte Carlo simulation"
```

---

## Task 10: Seed model — delta-hedged long option on SPY

**Files:**
- Create: `models/options/01-delta-hedged-long-option-spy/backtest.py`
- Test: `tests/test_options_seed_model.py`

The strategy: hold a long ATM option, delta-hedge to zero each bar, roll a new option whenever none is live. `run_for_gui` returns the historical-path `BacktestResult`, the Monte Carlo `SimulationResult`, and a payoff curve. The test imports the module by path (slugs are not importable) the same way `app/adapters.py` does.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_options_seed_model.py
"""Smoke test for the delta-hedged-long-option seed model's GUI surface."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tradinglib.backtest import BacktestResult
from tradinglib.data.paths import repo_root


def _load_model_module():
    model_dir = repo_root() / "models/options/01-delta-hedged-long-option-spy"
    spec = importlib.util.spec_from_file_location("_seed_delta_hedge", model_dir / "backtest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_for_gui_returns_expected_keys() -> None:
    module = _load_model_module()
    out = module.run_for_gui("2023-01-01", "2023-06-30", symbol="SPY", n_paths=200)
    assert isinstance(out["result"], BacktestResult)
    assert "close" in out["data"].columns
    assert out["simulation"].pnl_distribution.shape[0] <= 200
    assert "payoff" in out
    assert {"spots", "values"} <= set(out["payoff"])
    assert out["symbol"] == "SPY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_options_seed_model.py -v`
Expected: FAIL — `backtest.py` does not exist yet (FileNotFoundError / spec is None).

- [ ] **Step 3: Implement the seed model**

```python
# models/options/01-delta-hedged-long-option-spy/backtest.py
"""Delta-hedged long option on SPY — the options 'hello-world'.

Buy a ~1-month ATM call, delta-hedge to zero with the underlying every bar,
and roll a fresh option whenever the previous one expires. Because the
position is continuously delta-hedged, its P&L isolates the gap between
*realized* volatility (how much SPY actually moved) and the *implied*
volatility we priced the option at — the cleanest exercise of the pricing +
Greeks machinery.

European exercise is used for the clean vol story; the American CRR pricer is
validated separately in the test suite and the notebook.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.options.instruments import CONTRACT_MULTIPLIER, OptionLeg
from tradinglib.options.pricing import bs_price
from tradinglib.options.simulate import run_simulation
from tradinglib.loaders.equities.yfinance import load_daily

SYMBOL = "SPY"
START = "2023-01-01"
END = "2024-12-31"
TENOR_DAYS = 30
IMPLIED_VOL = 0.18
RATE = 0.04
FEE_BPS = 1.0
SLIPPAGE_BPS = 0.5


class DeltaHedgedLongOption:
    """Hold one long ATM call, delta-hedge to zero each bar, roll at expiry."""

    def __init__(self, tenor_days: int = TENOR_DAYS, right: str = "call") -> None:
        self.tenor_days = tenor_days
        self.right = right

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        if not engine.position.legs:
            expiry = t + pd.Timedelta(days=self.tenor_days)
            engine.add_leg(OptionLeg(self.right, strike=round(spot), expiry=expiry, quantity=1.0))
        engine.hedge_to_delta(0.0)


def _payoff_curve(spot: float, strike: float, vol: float, rate: float, tenor_days: int) -> dict[str, Any]:
    """Value of one long call vs spot, today and at expiry — for the GUI plot."""
    spots = np.linspace(spot * 0.8, spot * 1.2, 80)
    t_yrs = tenor_days / 365.0
    today = np.array([bs_price("call", s, strike, t_yrs, vol, rate) * CONTRACT_MULTIPLIER for s in spots])
    at_expiry = np.maximum(spots - strike, 0.0) * CONTRACT_MULTIPLIER
    return {"spots": spots, "values": today, "expiry_values": at_expiry, "strike": strike}


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    implied_vol: float = IMPLIED_VOL,
    tenor_days: int = TENOR_DAYS,
    n_paths: int = 2_000,
) -> dict[str, Any]:
    """Run the historical backtest + Monte Carlo simulation without writing to disk."""
    bars = load_daily(symbol, start=start, end=end)
    prices = bars["close"]

    result = run_options_backtest(
        prices,
        DeltaHedgedLongOption(tenor_days=tenor_days),
        vol=implied_vol,
        rate=RATE,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )

    def factory() -> DeltaHedgedLongOption:
        return DeltaHedgedLongOption(tenor_days=tenor_days)

    spot0 = float(prices.iloc[0])
    simulation = run_simulation(
        factory,
        spot=spot0,
        vol=implied_vol,
        rate=RATE,
        days=min(tenor_days, len(prices)),
        n_paths=n_paths,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        seed=0,
    )

    data = pd.DataFrame({"close": prices, "delta_fraction": result.position})
    payoff = _payoff_curve(spot0, round(spot0), implied_vol, RATE, tenor_days)
    return {
        "data": data,
        "result": result,
        "simulation": simulation,
        "payoff": payoff,
        "symbol": symbol,
        "params": {
            "start": str(start),
            "end": str(end),
            "symbol": symbol,
            "implied_vol": implied_vol,
            "tenor_days": tenor_days,
            "n_paths": n_paths,
        },
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "results"
    out_dir.mkdir(exist_ok=True)

    out = run_for_gui()
    result = out["result"]

    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Delta-hedged option")
    ax.set_title("Delta-hedged long option on SPY")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.savefig(out_dir / "equity_curve.png", dpi=120, bbox_inches="tight")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_options_seed_model.py -v`
Expected: PASS (note: this downloads SPY data on first run; subsequent runs hit the yfinance cache).

- [ ] **Step 5: Generate the on-disk results and commit**

```bash
uv run python models/options/01-delta-hedged-long-option-spy/backtest.py
git add models/options/01-delta-hedged-long-option-spy/backtest.py models/options/01-delta-hedged-long-option-spy/results tests/test_options_seed_model.py
git commit -m "feat(options): delta-hedged long option seed model"
```

---

## Task 11: Model metadata + docs

**Files:**
- Create: `models/options/01-delta-hedged-long-option-spy/model.md`
- Create: `models/options/01-delta-hedged-long-option-spy/README.md`
- Modify: `MODELS.md` (regenerated)
- Modify: `docs/methodology.md`
- Modify: `README.md`

- [ ] **Step 1: Write `model.md`** (fill `sharpe_oos`/`max_drawdown` from the `results/metrics.json` produced in Task 10)

```markdown
---
name: Delta-Hedged Long Option on SPY
family: options
window: swing
assets: [equities]
data_sources: [yfinance_daily_bars]
tickers: any
default_ticker: SPY
status: working
sharpe_oos: <fill from results/metrics.json "sharpe">
max_drawdown: <fill from results/metrics.json "max_drawdown">
---

Buy a ~1-month ATM call on SPY and delta-hedge to zero every bar, rolling a
fresh option at expiry. Continuous hedging strips out directional exposure so
the P&L isolates realized-vs-implied volatility. Doubles as the options
pipeline's end-to-end smoke test: pricing, Greeks, expiry-roll, mark-to-market
accounting, and a Monte Carlo outcome distribution. See [`README.md`](README.md).
```

- [ ] **Step 2: Write the model `README.md`**

```markdown
# Delta-Hedged Long Option on SPY

The options family's hello-world. Holds a long ~1-month ATM SPY call,
delta-hedged to zero each bar with the underlying, rolled at expiry.

## What it tests

Continuous delta-hedging removes first-order directional exposure, so the
residual P&L is the gap between **realized** volatility (how much SPY actually
moved) and the **implied** volatility the option was priced at. It exercises
the whole options stack end-to-end: Black-Scholes pricing & Greeks
(`tradinglib.options.pricing`), the multi-leg position model
(`tradinglib.options.instruments`), the mark-to-market engine
(`tradinglib.backtest.options_engine`), and the GBM Monte Carlo simulation
(`tradinglib.options.simulate`).

## Caveats

- Constant implied-vol assumption — there is no vol surface or term structure
  in phase 1. The historical-chain loader (future phase) will replace this.
- European exercise. The American CRR pricer exists and is tested, but the
  clean vol story uses European pricing.
- Daily rehedge only; intraday gamma P&L is not captured.

Reproduce:

```bash
uv run python models/options/01-delta-hedged-long-option-spy/backtest.py
```
```

- [ ] **Step 3: Regenerate `MODELS.md`**

Run: `uv run python scripts/regenerate_models_index.py`
Expected: `MODELS.md` now lists the new options model. Verify with `git diff MODELS.md`.

- [ ] **Step 4: Update `docs/methodology.md`** — append this subsection after the "Execution model" section:

```markdown
## Options (mark-to-market) results

Options strategies run through `tradinglib.backtest.options_engine`, which
marks a multi-leg position to market each bar rather than using the linear
`position × return` math. The resulting `BacktestResult` reuses
`compute_metrics`, but two fields are reinterpreted:

- **`position`** — net portfolio *delta* expressed as a fraction of equity
  (`net_delta_shares × spot / equity`), not a target weight.
- **`turnover`** — traded notional (underlying + option premium) divided by
  equity for that bar.

`equity_curve` is the portfolio's mark-to-market value and `returns` is its
bar-over-bar percent change, so Sharpe/Sortino/drawdown stay comparable to
every other model.
```

- [ ] **Step 5: Update `README.md`** — add a row to the "Current models" table and a row to the "Repository tour" table:

Current-models table row (use the actual metrics from `results/metrics.json`):
```markdown
| [Delta-Hedged Long Option on SPY](models/options/01-delta-hedged-long-option-spy/) | options | swing | equities | <sharpe> | <dd> | working |
```

Repository-tour table row:
```markdown
| `models/options/` | Options pricing, Greeks, multi-leg payoffs, vol strategies |
```

- [ ] **Step 6: Commit**

```bash
git add models/options/01-delta-hedged-long-option-spy/model.md models/options/01-delta-hedged-long-option-spy/README.md MODELS.md docs/methodology.md README.md
git commit -m "docs(options): seed model metadata, methodology, and index"
```

---

## Task 12: Streamlit GUI integration

**Files:**
- Create: `app/ui/options_view.py`
- Modify: `app/streamlit_app.py:188-191` (the render block at the end)
- Test: manual (Streamlit UI)

The model is already discoverable (it has a `model.md`) and runnable (it has `run_for_gui`). This task adds the options-specific visuals: the payoff curve and the Monte Carlo P&L histogram, rendered only when `run_for_gui` provided those keys (so existing models are unaffected).

- [ ] **Step 1: Implement `options_view.py`**

```python
# app/ui/options_view.py
"""Options-specific panels: payoff curve and Monte Carlo P&L distribution.

Rendered only when a model's run_for_gui output carries the optional
``payoff`` / ``simulation`` keys, so non-options models are unaffected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render(out: dict[str, Any]) -> None:
    if "payoff" in out:
        _render_payoff(out["payoff"])
    if "simulation" in out:
        _render_simulation(out["simulation"])


def _render_payoff(payoff: dict[str, Any]) -> None:
    st.subheader("Option payoff (single leg)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=payoff["spots"], y=payoff["values"], name="Value today"))
    fig.add_trace(
        go.Scatter(x=payoff["spots"], y=payoff["expiry_values"], name="Value at expiry",
                   line={"dash": "dash"})
    )
    fig.add_vline(x=payoff["strike"], line_dash="dot", line_color="gray")
    fig.update_layout(height=320, margin={"t": 30, "b": 30, "l": 0, "r": 0},
                      xaxis_title="Underlying spot", yaxis_title="Position value ($)")
    st.plotly_chart(fig, use_container_width=True)


def _render_simulation(sim: Any) -> None:
    st.subheader("Monte Carlo P&L distribution")
    if getattr(sim, "truncated", False):
        st.caption("⚠️ path count was capped to bound memory; showing the capped sample.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prob. of profit", f"{sim.prob_of_profit:.0%}")
    c2.metric("Median P&L", f"${sim.percentiles[50]:,.0f}")
    c3.metric("Mean P&L", f"${sim.mean:,.0f}")
    c4.metric("Expected shortfall (5%)", f"${sim.expected_shortfall:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim.pnl_distribution, nbinsx=60, name="P&L"))
    for level in (5, 50, 95):
        fig.add_vline(x=sim.percentiles[level], line_dash="dot",
                      annotation_text=f"p{level}", line_color="gray")
    fig.update_layout(height=340, margin={"t": 30, "b": 30, "l": 0, "r": 0},
                      xaxis_title="Terminal P&L ($)", yaxis_title="Paths")
    st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Wire it into `app/streamlit_app.py`**

Find the import block near the top (currently):
```python
from app.ui import data_details, data_view, results_view
```
Replace with:
```python
from app.ui import data_details, data_view, options_view, results_view
```

Find the final render block (currently the last three lines of the file):
```python
data_details.render(selected, out)
data_view.render(selected, out)
results_view.render(out)
```
Replace with:
```python
data_details.render(selected, out)
data_view.render(selected, out)
results_view.render(out)
options_view.render(out)  # no-op unless the model returned payoff/simulation
```

- [ ] **Step 3: Manually verify in the app**

Run: `uv run streamlit run app/streamlit_app.py`
Steps: select **Delta-Hedged Long Option on SPY**, pick a 2023 date range, press **Run backtest**.
Expected: the standard equity curve + metrics render, followed by the option payoff curve and a Monte Carlo P&L histogram with p5/p50/p95 markers. Selecting any non-options model (e.g. SMA Crossover) renders exactly as before (no options panels).

- [ ] **Step 4: Commit**

```bash
git add app/ui/options_view.py app/streamlit_app.py
git commit -m "feat(options): Streamlit payoff + Monte Carlo panels"
```

---

## Task 13: Full test suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -q`
Expected: all tests pass, including the pre-existing suite and the new options tests.

- [ ] **Step 2: Lint and type-check** (matches the repo's dev tooling)

Run: `uv run ruff check tradinglib app models tests` and `uv run mypy tradinglib`
Expected: no errors. Fix any reported issues and re-run.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore(options): lint and type-check fixes"
```

---

## Self-Review Notes

- **Spec coverage:** pricing (Tasks 2–5), instruments (6), engine + result reinterpretation (7), GBM + outcome simulation (8–9), seed delta-hedged model with historical + MC views (10), new `options/` family + docs (11), GUI integration with payoff + MC histogram (12). All spec sections map to a task.
- **American exercise:** implemented and tested in Task 5; exercised by the engine's `_price_leg` for `style="american"` legs. The seed model uses European for the clean vol story (per spec).
- **Memory:** `gbm_paths` is float32, `run_simulation` aggregates to per-path P&L and caps `n_paths` via `max_paths` with a surfaced `truncated` flag (spec's Streamlit memory constraint).
- **Type consistency:** `run_options_backtest`, `OptionsEngine`, `OptionLeg`, `Position`, `SimulationResult`, `gbm_paths`, `run_simulation`, `run_for_gui` signatures are identical everywhere they appear across tasks. `CONTRACT_MULTIPLIER` is defined once in `instruments.py` and imported elsewhere.
- **Backward compatibility:** GUI options panels render only when `payoff`/`simulation` keys are present, so existing models are untouched.
```
