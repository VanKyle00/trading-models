"""Structure builders: payoff math, PoP vs hand-computed lognormal, gate fallbacks."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy.stats import norm

from tradinglib.strategist.structures import (
    build_structures,
    pop_market_implied,
    stock_plan,
)
from tradinglib.tournament.levels import Levels

ASOF = pd.Timestamp("2026-06-10")
LONG_LEVELS = Levels(entry=100.0, entry_type="market", stop=96.0, target=108.0, condition="t")
SHORT_LEVELS = Levels(entry=100.0, entry_type="market", stop=104.0, target=92.0, condition="t")

# (right, dte, strike) -> mid. Engineered so: 75-DTE call delta~0.65 -> K95;
# CSP/bull-put strikes sit under the 96 stop with credit >= width/3.
LONG_MIDS = {
    ("call", 75, 95.0): 8.0,
    ("call", 75, 100.0): 5.5,
    ("call", 75, 105.0): 3.6,
    ("call", 75, 110.0): 2.4,
    ("put", 38, 85.0): 0.6,
    ("put", 38, 90.0): 1.0,
    ("put", 38, 95.0): 2.8,
}
SHORT_MIDS = {
    ("put", 75, 90.0): 1.8,
    ("put", 75, 95.0): 2.6,
    ("put", 75, 105.0): 7.5,
    ("put", 75, 110.0): 11.0,
    ("call", 38, 105.0): 2.4,
    ("call", 38, 110.0): 0.7,
}


def test_pop_market_implied_matches_hand_computed_lognormal() -> None:
    # P(S_T < 110 | S0=100, vol=0.2, T=0.25), zero-rate lognormal:
    d = (math.log(110.0 / 100.0) + 0.5 * 0.2**2 * 0.25) / (0.2 * math.sqrt(0.25))
    p_below = float(norm.cdf(d))

    above = pop_market_implied(spot=100.0, level=110.0, vol=0.2, t_years=0.25, profit_above=True)
    below = pop_market_implied(spot=100.0, level=110.0, vol=0.2, t_years=0.25, profit_above=False)

    assert below == pytest.approx(p_below)
    assert above == pytest.approx(1.0 - p_below)
    assert below == pytest.approx(0.842, abs=1e-3)  # sanity anchor


def test_stock_plan_long() -> None:
    s = stock_plan(LONG_LEVELS, "long")

    assert s.kind == "stock" and s.unit == "share" and s.legs == []
    assert s.max_loss == pytest.approx(4.0)  # entry-to-stop plan risk
    assert s.max_gain == pytest.approx(8.0)
    assert s.breakeven == pytest.approx(100.0)
    assert s.rr == pytest.approx(2.0)
    assert s.pop_market_implied is None and s.premium_yield is None
    assert s.warnings == []


def test_stock_plan_short_carries_borrow_warning() -> None:
    s = stock_plan(SHORT_LEVELS, "short")

    assert s.kind == "stock_short"
    assert s.max_loss == pytest.approx(4.0) and s.max_gain == pytest.approx(8.0)
    assert any("borrow" in w for w in s.warnings)
