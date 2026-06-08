"""Tests for the EarningsStraddle options strategy (timing, frictions, cap)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
        earnings_datetime=earnings,
        entry_lead=3,
        exit_offset=1,
        contracts=1.0,
        bar_index=prices.index,
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
        earnings_datetime=earnings,
        entry_lead=3,
        exit_offset=1,
        contracts=1.0,
        bar_index=prices.index,
    )
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    run_options_backtest(prices, s, surface=surface, spread=NoSpread())

    assert s.entered_on is not None
    assert s.exited_on is not None
    assert s.exited_on > s.entered_on
    # Explicit no-pre-settle: the leg expiry (earnings + 14 calendar days) is
    # strictly after the exit bar, so engine._settle_expiries (which fires when
    # (expiry - t).days <= 0) provably cannot have settled the leg first. exited_on
    # being set therefore proves the close ran via close_all_options, not a silent
    # pre-settlement.
    expiry = pd.Timestamp(earnings) + pd.Timedelta(days=14)
    assert (expiry - pd.Timestamp(s.exited_on)).days > 0


def test_entry_counts_actual_bars_not_business_days() -> None:
    # Gapped calendar: drop the two bars immediately before the earnings bar so
    # that counting back `entry_lead` ACTUAL price bars lands on a different date
    # than `earnings - BDay(entry_lead)`. The plan specifies entry at the Nth
    # actual bar before earnings, so entered_on must be the bar `entry_lead`
    # positions before the earnings bar in the gapped index, not a calendar BDay.
    full = pd.date_range("2024-02-01", periods=20, freq="B")
    earnings = full[10]
    idx = pd.DatetimeIndex([d for i, d in enumerate(full) if i not in (8, 9)])
    prices = pd.Series(100.0, index=idx, name="close")
    s = strat.EarningsStraddle(
        earnings_datetime=earnings,
        entry_lead=3,
        exit_offset=1,
        contracts=1.0,
        bar_index=prices.index,
    )
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)
    run_options_backtest(prices, s, surface=surface, spread=NoSpread())

    e_idx = idx.get_loc(earnings)
    assert s.entered_on == idx[e_idx - 3]  # 3 actual bars before earnings, not BDay
    assert s.exited_on == idx[e_idx + 1]  # 1 actual bar after earnings


def test_bar_index_is_required() -> None:
    # The strategy plans entry/exit from the bar schedule up front (the entry bar
    # precedes the earnings bar), so bar_index is mandatory. Omitting it must raise
    # rather than silently degrade to a business-day approximation.
    earnings = _flat_prices().index[10]
    with pytest.raises(TypeError):
        strat.EarningsStraddle(earnings_datetime=earnings, entry_lead=3, exit_offset=1)


def test_straddle_pays_spread_on_both_legs() -> None:
    prices = _flat_prices()
    earnings = prices.index[10]
    # non-zero pre_iv and ATM-snapped strike => the straddle carries real premium,
    # so the per-leg half-spread is unambiguously charged (entry x2, exit x2).
    surface = EventVolSurface(earnings_datetime=earnings, pre_iv=0.60, post_iv=0.30)

    def _run(spread):
        s = strat.EarningsStraddle(
            earnings_datetime=earnings,
            entry_lead=3,
            exit_offset=1,
            contracts=1.0,
            bar_index=prices.index,
        )
        return run_options_backtest(prices, s, surface=surface, spread=spread)

    frictionless = _run(NoSpread())
    frictioned = _run(ParametricSpread())

    assert frictioned.equity_curve.iloc[-1] < frictionless.equity_curve.iloc[-1]
    assert (frictioned.turnover > 0).sum() >= 2


def test_size_contracts_uses_fixed_fraction_of_capital() -> None:
    # 1% of $100k = $1000 budget; premium $5.00/share * 100 = $500/contract -> 2
    n = strat.size_contracts(capital=100_000.0, risk_fraction=0.01, straddle_premium=5.0)
    assert n == 2.0


def test_size_contracts_zero_when_premium_exceeds_budget() -> None:
    n = strat.size_contracts(capital=100_000.0, risk_fraction=0.001, straddle_premium=50.0)
    assert n == 0.0


def test_portfolio_cap_blocks_when_at_capacity() -> None:
    assert strat.can_open(open_count=2, max_concurrent=2) is False
    assert strat.can_open(open_count=1, max_concurrent=2) is True
    assert strat.can_open(open_count=0, max_concurrent=1) is True


def _load_backtest():
    import importlib.util as _ilu

    bt_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "options"
        / "03-earnings-straddle-spy"
        / "backtest.py"
    )
    _bspec = _ilu.spec_from_file_location("earnings_backtest", bt_path)
    assert _bspec is not None and _bspec.loader is not None
    bt = _ilu.module_from_spec(_bspec)
    _bspec.loader.exec_module(bt)
    return bt


def test_run_synthetic_returns_filtered_and_unfiltered_pnl_tz_aware_earnings() -> None:
    bt = _load_backtest()
    # tz-NAIVE price index (like a tz-stripped load_daily) ...
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.005, 60)), index=idx)
    close.iloc[31:] *= 1.15  # big post-earnings move
    # ... and a tz-AWARE earnings timestamp (like the loader emits)
    earnings = pd.Timestamp(idx[30], tz="UTC")

    report = bt.run_synthetic(
        close=close,
        earnings_datetime=earnings,
        pre_iv=0.45,
        post_iv=0.25,
        k=1.2,
        lookback=4,
        past_moves=[0.12, 0.10, 0.14, 0.11],  # prior events only (no leakage)
    )

    assert "filtered" in report and "unfiltered" in report
    assert "metrics" in report["filtered"]
    assert "took_trade" in report["filtered"]
    assert "final_equity" in report["filtered"]
    assert "final_equity" in report["unfiltered"]
    assert "trade_pnl" in report["filtered"]


def test_run_synthetic_rejects_earnings_out_of_range() -> None:
    bt = _load_backtest()
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    close = pd.Series(100.0, index=idx)
    with pytest.raises(ValueError):
        bt.run_synthetic(
            close=close,
            earnings_datetime=pd.Timestamp("2025-01-01"),  # after last bar
            pre_iv=0.45,
            post_iv=0.25,
            past_moves=[0.10],
        )
    with pytest.raises(ValueError):
        bt.run_synthetic(
            close=close,
            earnings_datetime=pd.Timestamp(idx[1]),  # e_idx - ENTRY_LEAD < 0
            pre_iv=0.45,
            post_iv=0.25,
            past_moves=[0.10],
        )


def test_build_validation_report_shapes() -> None:
    bt = _load_backtest()
    # two tickers, each with two synthetic per-event filtered trade P&Ls
    per_ticker_pnl = {
        "AAA": [120.0, -40.0],
        "BBB": [-10.0, -20.0],
    }
    branches = {
        "AAA": {"filtered": {"final_equity": 100120.0}, "unfiltered": {"final_equity": 100050.0}},
        "BBB": {"filtered": {"final_equity": 99970.0}, "unfiltered": {"final_equity": 99990.0}},
    }
    report = bt.build_validation_report(branches=branches, per_ticker_pnl=per_ticker_pnl)

    assert "pooled_filtered" in report
    assert {
        "bootstrap_t_stat",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_p_value",
    } <= report["pooled_filtered"].keys()
    assert "fdr" in report
    assert report["fdr"]["tickers"] == ["AAA", "BBB"]
    assert len(report["fdr"]["p_values"]) == 2
    assert len(report["fdr"]["rejected"]) == 2
    assert "trade_metrics" in report
    assert report["trade_metrics"]["n_trades"] == 4


def test_plot_branches_writes_png(tmp_path) -> None:
    bt = _load_backtest()
    branches = {
        "AAA": {"filtered": {"final_equity": 100120.0}, "unfiltered": {"final_equity": 100050.0}},
    }
    out = tmp_path / "equity_curve.png"
    bt.plot_branches(branches, out)
    assert out.exists() and out.stat().st_size > 0


def test_run_for_gui_satisfies_service_contract() -> None:
    # The service layer (tradinglib/service/run.py) runs every registered model
    # and requires run_for_gui to return a BacktestResult under "result" plus a
    # price DataFrame with a "close" column under "data". Mock the loaders so the
    # contract is exercised offline/deterministically.
    bt = _load_backtest()
    idx = pd.date_range("2023-01-02", periods=80, freq="B", tz="UTC")
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.005, 80)), index=idx)
    bars = pd.DataFrame({"close": close})
    e_dates = pd.Series([idx[20], idx[50]])  # two events; second is "traded"

    bt.load_daily = lambda symbol, start=None, end=None, **kw: bars
    bt.get_earnings_dates = lambda tickers, start=None, end=None, **kw: pd.DataFrame(
        {"earnings_datetime": e_dates}
    )

    out = bt.run_for_gui("2023-01-01", "2024-01-01", symbol="SPY")

    from tradinglib.backtest.engine import BacktestResult

    assert isinstance(out["result"], BacktestResult)
    assert "close" in out["data"].columns
    assert out["symbol"] == "SPY"
    assert isinstance(out["params"], dict)
    assert out["report"] is not None
