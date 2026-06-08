"""Tests for the EarningsStraddle options strategy (timing, frictions, cap)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from tradinglib.backtest.options_engine import run_options_backtest
from tradinglib.options.spread import NoSpread, ParametricSpread
from tradinglib.options.surface import EventVolSurface

_STRAT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "options"
    / "03-earnings-straddle-spy"
    / "strategy.py"
)
_spec = importlib.util.spec_from_file_location("earnings_strategy", _STRAT_PATH)
assert _spec is not None and _spec.loader is not None
strat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strat)


def _flat_prices(n: int = 20, start: str = "2024-02-01") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(100.0, index=idx, name="close")


def test_enters_at_lead_and_exits_at_offset() -> None:
    prices = _flat_prices()
    earnings = prices.index[10]
    s = strat.EarningsStraddle(
        earnings_datetime=earnings, entry_lead=3, exit_offset=1, contracts=1.0
    )
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    run_options_backtest(prices, s, surface=surface, spread=NoSpread())

    assert s.entered_on == prices.index[7]
    assert s.exited_on == prices.index[11]


def test_friday_earnings_exit_is_closed_not_auto_settled() -> None:
    # earnings on a Friday; exit is the next business bar (Monday). With the
    # 14-day default tenor the expiry is far beyond exit, so close_all_options
    # handles it and exited_on is set (no pre-emptive _settle_expiries).
    prices = _flat_prices(n=25, start="2024-02-05")  # 2024-02-05 is a Monday
    fridays = [d for d in prices.index if d.dayofweek == 4]
    earnings = fridays[1]
    s = strat.EarningsStraddle(
        earnings_datetime=earnings, entry_lead=3, exit_offset=1, contracts=1.0
    )
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    run_options_backtest(prices, s, surface=surface, spread=NoSpread())

    assert s.entered_on is not None
    assert s.exited_on is not None
    assert s.exited_on > s.entered_on


def test_straddle_pays_spread_on_both_legs() -> None:
    prices = _flat_prices()
    earnings = prices.index[10]
    # non-zero pre_iv and ATM-snapped strike => the straddle carries real premium,
    # so the per-leg half-spread is unambiguously charged (entry x2, exit x2).
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)

    def _run(spread):
        s = strat.EarningsStraddle(
            earnings_datetime=earnings, entry_lead=3, exit_offset=1, contracts=1.0
        )
        return run_options_backtest(prices, s, surface=surface, spread=spread)

    frictionless = _run(NoSpread())
    frictioned = _run(ParametricSpread())

    assert frictioned.equity_curve.iloc[-1] < frictionless.equity_curve.iloc[-1]
    assert (frictioned.turnover > 0).sum() >= 2
