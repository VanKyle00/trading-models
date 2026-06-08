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
        prices,
        _OpenThenClose(expiry),
        surface=surface,
        spread=ParametricSpread(),
        fee_bps=0,
        slippage_bps=0,
    )
    res0 = run_options_backtest(
        prices,
        _OpenThenClose(expiry),
        surface=surface,
        spread=NoSpread(),
        fee_bps=0,
        slippage_bps=0,
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


def test_config_records_resolved_models() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.Series(100.0, index=idx)
    # Default spread path resolves to NoSpread — config must record that, not 'NoneType'.
    res = run_options_backtest(prices, _DoNothing(), surface=FlatSurface(0.2))
    assert res.config["surface"] == "FlatSurface"
    assert res.config["spread"] == "NoSpread"
    explicit = run_options_backtest(
        prices, _DoNothing(), surface=FlatSurface(0.2), spread=ParametricSpread()
    )
    assert explicit.config["spread"] == "ParametricSpread"
