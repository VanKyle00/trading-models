"""Tests for ATM straddle assembly helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.options.pricing import bs_price
from tradinglib.options.straddle import atm_straddle_legs, snap_strike, straddle_price


def test_atm_straddle_legs_snaps_strike_and_builds_two_legs() -> None:
    expiry = pd.Timestamp("2024-02-01")
    legs = atm_straddle_legs(spot=101.4, expiry=expiry, quantity=1.0, strike_step=1.0)

    assert len(legs) == 2
    assert sorted(leg.right for leg in legs) == ["call", "put"]
    assert {leg.strike for leg in legs} == {101.0}
    assert all(leg.expiry == expiry for leg in legs)
    assert all(leg.quantity == 1.0 for leg in legs)


def test_atm_straddle_legs_strike_step_5() -> None:
    legs = atm_straddle_legs(
        spot=103.0, expiry=pd.Timestamp("2024-02-01"), quantity=1.0, strike_step=5.0
    )
    assert {leg.strike for leg in legs} == {105.0}


def test_snap_strike_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError):
        snap_strike(100.0, 0.0)


def test_straddle_price_matches_two_bs_legs() -> None:
    expected = bs_price("call", 100.0, 100.0, 30 / 365.0, 0.40, 0.04) + bs_price(
        "put", 100.0, 100.0, 30 / 365.0, 0.40, 0.04
    )
    price = straddle_price(spot=100.0, strike=100.0, t=30 / 365.0, vol=0.40, rate=0.04)
    assert price == pytest.approx(expected, abs=1e-9)
    assert price > 0.0
