"""build_exit_plan: BSM-repriced triggers, threshold math, time rules, notes."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.options.pricing import bs_price
from tradinglib.strategist.exit_plan import build_exit_plan
from tradinglib.strategist.structures import Band, Structure, stock_plan
from tradinglib.tournament.levels import Levels

ASOF = pd.Timestamp("2026-06-10")
LONG_LEVELS = Levels(entry=100.0, entry_type="market", stop=96.0, target=108.0, condition="t")


def _structure(
    *, legs: list[dict], premium: float, max_gain: float | None, quantity: int | None
) -> Structure:
    return Structure(
        kind="x",
        label="x",
        unit="contract",
        legs=legs,
        premium=premium,
        max_loss=abs(premium),
        max_gain=max_gain,
        breakeven=None,
        pop_market_implied=None,
        rr=None,
        premium_yield=None,
        quantity=quantity,
    )


def _leg(
    action: str, right: str, strike: float, *, dte: int = 75, iv: float = 0.30, mid: float = 1.0
) -> dict:
    return {
        "action": action,
        "right": right,
        "strike": strike,
        "expiration": (ASOF + pd.Timedelta(days=dte)).strftime("%Y-%m-%d"),
        "dte": dte,
        "mid": mid,
        "iv": iv,
        "delta": 0.5,
    }


def test_long_call_price_rules_reprice_at_half_dte() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=8.0)], premium=8.0, max_gain=None, quantity=2
    )
    plan = build_exit_plan(s, LONG_LEVELS)

    assert plan is not None
    # 75 DTE -> evaluated with 38 days left, at the user's own target/stop spots
    target_value = bs_price("call", 108.0, 95.0, 38 / 365, 0.30, 0.0)
    stop_value = bs_price("call", 96.0, 95.0, 38 / 365, 0.30, 0.0)
    tp, cut = plan.price_rules
    assert tp.action == "take profit" and "108" in tp.trigger
    assert tp.est_value == pytest.approx(round(target_value, 2))
    assert tp.est_pnl == pytest.approx(round((target_value - 8.0) * 200, 0))  # 2 contracts
    assert tp.est_pnl_pct == pytest.approx(round((target_value - 8.0) / 8.0, 2))
    assert cut.action == "close" and "96" in cut.trigger
    assert cut.est_pnl == pytest.approx(round((stop_value - 8.0) * 200, 0))
    assert plan.est_by == (ASOF + pd.Timedelta(days=37)).strftime("%Y-%m-%d")


def test_debit_thresholds_and_time_stop() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=8.0)], premium=8.0, max_gain=None, quantity=1
    )
    plan = build_exit_plan(s, LONG_LEVELS)

    assert plan is not None
    assert "$16.00" in plan.profit_take.trigger  # +100% of the 8.0 debit
    assert plan.profit_take.est_pnl == pytest.approx(800.0)
    assert plan.profit_take.est_pnl_pct == 1.0
    assert "$4.00" in plan.loss_cut.trigger  # -50% of the debit
    assert plan.loss_cut.est_pnl == pytest.approx(-400.0)
    # debit time stop: final third of 75 DTE = 25 days before expiry
    (ts,) = plan.time_rules
    assert ts.action.startswith("exit")
    assert (ASOF + pd.Timedelta(days=50)).strftime("%Y-%m-%d") in ts.trigger


def test_credit_thresholds_and_21_dte_rule() -> None:
    legs = [_leg("sell", "put", 95.0, dte=38, mid=2.8), _leg("buy", "put", 90.0, dte=38, mid=1.0)]
    s = _structure(legs=legs, premium=-1.8, max_gain=1.8, quantity=1)
    plan = build_exit_plan(s, LONG_LEVELS)

    assert plan is not None
    assert "$0.90" in plan.profit_take.trigger  # buy back at 50% of the 1.80 credit
    assert plan.profit_take.est_pnl == pytest.approx(90.0)
    assert plan.profit_take.est_pnl_pct == 0.5
    assert "$3.60" in plan.loss_cut.trigger  # buyback at 2x credit = credit lost
    assert plan.loss_cut.est_pnl == pytest.approx(-180.0)
    (roll,) = plan.time_rules
    assert "21 DTE" in roll.trigger and roll.action == "close or roll"
    assert (ASOF + pd.Timedelta(days=17)).strftime("%Y-%m-%d") in roll.trigger


def test_21_dte_rule_omitted_inside_21_dte() -> None:
    legs = [_leg("sell", "put", 95.0, dte=15, mid=2.8), _leg("buy", "put", 90.0, dte=15, mid=1.0)]
    s = _structure(legs=legs, premium=-1.8, max_gain=1.8, quantity=1)
    plan = build_exit_plan(s, LONG_LEVELS)
    assert plan is not None and plan.time_rules == []


def _condor_value(spot: float) -> float:
    t = 19 / 365
    return (
        bs_price("put", spot, 85.0, t, 0.30, 0.0)
        - bs_price("put", spot, 90.0, t, 0.30, 0.0)
        - bs_price("call", spot, 110.0, t, 0.30, 0.0)
        + bs_price("call", spot, 115.0, t, 0.30, 0.0)
    )


def test_neutral_band_rules_close_at_both_edges() -> None:
    legs = [
        _leg("buy", "put", 85.0, dte=38, mid=0.6),
        _leg("sell", "put", 90.0, dte=38, mid=1.6),
        _leg("sell", "call", 110.0, dte=38, mid=1.5),
        _leg("buy", "call", 115.0, dte=38, mid=0.5),
    ]
    s = _structure(legs=legs, premium=-2.0, max_gain=2.0, quantity=1)
    plan = build_exit_plan(s, Band(lower=92.0, upper=108.0, condition="t"))

    assert plan is not None
    lo, hi = plan.price_rules
    assert "92" in lo.trigger and lo.action == "close"
    assert "108" in hi.trigger and hi.action == "close"
    v_floor, v_ceil = _condor_value(92.0), _condor_value(108.0)
    assert v_floor < 0  # net short: closing the condor costs money (sign anchor)
    assert lo.est_value == pytest.approx(round(v_floor, 2))
    assert lo.est_pnl == pytest.approx(round((v_floor + 2.0) * 100, 0))
    assert lo.est_pnl_pct == pytest.approx(round((v_floor + 2.0) / 2.0, 2))
    assert hi.est_value == pytest.approx(round(v_ceil, 2))
    assert hi.est_pnl == pytest.approx(round((v_ceil + 2.0) * 100, 0))


def test_earnings_note_on_expiry_day_intraday_timestamp() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=8.0)], premium=8.0, max_gain=None, quantity=1
    )
    expiry_day_am = pd.Timestamp(
        (ASOF + pd.Timedelta(days=75)).strftime("%Y-%m-%d") + " 11:00", tz="UTC"
    )
    plan = build_exit_plan(s, LONG_LEVELS, next_earnings=expiry_day_am)
    assert plan is not None and len(plan.notes) == 1


def test_zero_premium_skips_threshold_and_time_rules() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=0.0)], premium=0.0, max_gain=None, quantity=1
    )
    plan = build_exit_plan(s, LONG_LEVELS)
    assert plan is not None
    assert plan.profit_take is None and plan.loss_cut is None and plan.time_rules == []
    assert all(r.est_pnl_pct is None for r in plan.price_rules)


def test_earnings_note_only_when_before_expiry() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=8.0)], premium=8.0, max_gain=None, quantity=1
    )

    inside = build_exit_plan(s, LONG_LEVELS, next_earnings=pd.Timestamp("2026-07-01", tz="UTC"))
    assert inside is not None and any("earnings 2026-07-01" in n for n in inside.notes)

    after = build_exit_plan(s, LONG_LEVELS, next_earnings=pd.Timestamp("2026-12-01", tz="UTC"))
    assert after is not None and after.notes == []


def test_unsized_uses_one_lot_and_stock_returns_none() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=8.0)], premium=8.0, max_gain=None, quantity=0
    )
    plan = build_exit_plan(s, LONG_LEVELS)
    assert plan is not None
    assert plan.profit_take.est_pnl == pytest.approx(800.0)  # 1-lot preview

    assert build_exit_plan(stock_plan(LONG_LEVELS, "long"), LONG_LEVELS) is None


def test_theta_week_copied_from_structure() -> None:
    s = _structure(
        legs=[_leg("buy", "call", 95.0, mid=8.0)], premium=8.0, max_gain=None, quantity=1
    )
    s.theta_week = -0.42
    plan = build_exit_plan(s, LONG_LEVELS)
    assert plan is not None and plan.theta_week == -0.42
