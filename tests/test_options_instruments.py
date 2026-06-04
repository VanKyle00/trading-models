"""Tests for the options instrument model."""

from __future__ import annotations

import dataclasses

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
    with pytest.raises(dataclasses.FrozenInstanceError):
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
    # short a put, spot 80 -> liability of intrinsic(20) * 100
    assert pos.intrinsic_value(spot=80.0) == pytest.approx(-20.0 * CONTRACT_MULTIPLIER)


def test_empty_position_is_zero() -> None:
    assert Position().intrinsic_value(spot=100.0) == 0.0


def test_multi_leg_straddle_intrinsic() -> None:
    call_leg = _leg(right="call", strike=100.0, quantity=1.0)
    put_leg = _leg(right="put", strike=100.0, quantity=1.0)
    pos = Position(legs=[call_leg, put_leg])
    # spot=120: call ITM by 20, put worthless -> 1 * 20 * 100 + 1 * 0 * 100 = 2000
    assert pos.intrinsic_value(spot=120.0) == pytest.approx(20.0 * CONTRACT_MULTIPLIER)
    # spot=80: put ITM by 20, call worthless -> 1 * 0 * 100 + 1 * 20 * 100 = 2000
    assert pos.intrinsic_value(spot=80.0) == pytest.approx(20.0 * CONTRACT_MULTIPLIER)
