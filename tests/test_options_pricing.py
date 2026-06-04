"""Tests for tradinglib.options pricing primitives."""

from __future__ import annotations

import math

import pytest


def test_package_imports() -> None:
    import tradinglib.options  # noqa: F401


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
