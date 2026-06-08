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
