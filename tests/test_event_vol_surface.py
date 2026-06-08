"""Tests for the synthetic pre/post-earnings event-vol surface."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.options.surface import EventVolSurface


def test_iv_elevated_before_and_crushed_after_earnings() -> None:
    earnings = pd.Timestamp("2024-02-15")
    surf = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    expiry = pd.Timestamp("2024-02-23")
    before = surf.iv(100.0, 100.0, expiry, pd.Timestamp("2024-02-14"))
    after = surf.iv(100.0, 100.0, expiry, pd.Timestamp("2024-02-16"))
    assert before == pytest.approx(0.60)
    assert after == pytest.approx(0.30)
    assert before > after


def test_iv_on_earnings_day_is_still_pre_crush() -> None:
    earnings = pd.Timestamp("2024-02-15")
    surf = EventVolSurface(earnings_datetime=earnings, pre_iv=0.55, post_iv=0.25)
    on_day = surf.iv(100.0, 100.0, pd.Timestamp("2024-02-23"), pd.Timestamp("2024-02-15"))
    assert on_day == pytest.approx(0.55)


def test_post_iv_must_be_below_pre_iv() -> None:
    # the entire point is IV crush; reject configs that manufacture an expansion
    with pytest.raises(ValueError):
        EventVolSurface(earnings_datetime=pd.Timestamp("2024-02-15"), pre_iv=0.30, post_iv=0.45)
    with pytest.raises(ValueError):
        EventVolSurface(earnings_datetime=pd.Timestamp("2024-02-15"), pre_iv=0.30, post_iv=0.0)


def test_iv_tz_robust_aware_earnings_naive_bar() -> None:
    # loader earnings_datetime is UTC-aware; engine bars can be tz-naive
    earnings = pd.Timestamp("2024-02-15", tz="UTC")
    surf = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    before = surf.iv(100.0, 100.0, pd.Timestamp("2024-02-23"), pd.Timestamp("2024-02-14"))
    after = surf.iv(100.0, 100.0, pd.Timestamp("2024-02-23"), pd.Timestamp("2024-02-16"))
    assert before == pytest.approx(0.60)
    assert after == pytest.approx(0.30)


def test_event_vol_surface_is_reexported() -> None:
    from tradinglib.options import EventVolSurface as Reexported
    from tradinglib.options.surface import EventVolSurface as Direct

    assert Reexported is Direct
