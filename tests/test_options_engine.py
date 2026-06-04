"""Tests for the options backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest import BacktestResult
from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.options.instruments import OptionLeg


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

    result = run_options_backtest(prices, BuyOnce(), vol=0.2, rate=0.04, fee_bps=0, slippage_bps=0)
    assert result.equity_curve.iloc[-1] == pytest.approx(result.equity_curve.iloc[-2], abs=1e-6)


def test_negative_equity_yields_nan_diagnostics() -> None:
    # Short a naked call into a large up-move so equity goes negative.
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.Series([100.0, 100.0, 5000.0, 5000.0, 5000.0], index=idx)
    expiry = idx[-1] + pd.Timedelta(days=30)

    class ShortCall:
        def __init__(self) -> None:
            self.opened = False

        def on_bar(self, engine: OptionsEngine, t, spot) -> None:
            if not self.opened:
                # Tiny capital, large short position -> guaranteed blow-up on the jump.
                engine.add_leg(OptionLeg("call", strike=100.0, expiry=expiry, quantity=-50.0))
                self.opened = True

    result = run_options_backtest(
        prices,
        ShortCall(),
        vol=0.2,
        rate=0.04,
        initial_capital=1_000.0,
        fee_bps=0,
        slippage_bps=0,
    )
    assert (result.equity_curve < 0).any()  # confirm the blow-up actually happened
    # On the blown-up bars, diagnostics are NaN rather than sign-flipped numbers.
    blown = result.equity_curve < 0
    assert np.isnan(result.position[blown]).all()
    assert np.isnan(result.turnover[blown]).all()


def test_delta_hedged_position_is_insensitive_to_small_moves() -> None:
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
    unhedged = run_options_backtest(
        prices, Unhedged(), vol=0.2, rate=0.04, fee_bps=0, slippage_bps=0
    )
    hedged_move = abs(hedged.equity_curve.iloc[1] - hedged.equity_curve.iloc[0])
    unhedged_move = abs(unhedged.equity_curve.iloc[1] - unhedged.equity_curve.iloc[0])
    assert hedged_move < unhedged_move
