"""Tests for tradinglib.options pricing primitives."""

from __future__ import annotations

import math

import pytest

from tradinglib.options.pricing import bs_greeks, bs_price, crr_price, implied_vol


def test_package_imports() -> None:
    import tradinglib.options  # noqa: F401


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


def test_implied_vol_round_trips() -> None:
    true_vol = 0.27
    price = bs_price("call", 100, 105, 0.5, true_vol, 0.04)
    assert implied_vol(price, "call", 100, 105, 0.5, 0.04) == pytest.approx(true_vol, abs=1e-4)


def test_implied_vol_put_round_trips() -> None:
    true_vol = 0.18
    price = bs_price("put", 100, 95, 0.75, true_vol, 0.04)
    assert implied_vol(price, "put", 100, 95, 0.75, 0.04) == pytest.approx(true_vol, abs=1e-4)


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
