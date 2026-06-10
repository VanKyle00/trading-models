"""Liquidity gate and strike/expiry pickers (synthetic chains, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.strategist.quotes import (
    LegQuote,
    at_or_below,
    atm_iv,
    by_delta,
    liquid_quotes,
    nearest_strike,
    pick_expiration,
    strictly_above,
    strictly_below,
)

ASOF = pd.Timestamp("2026-06-10")


def _q(strike: float, *, delta: float = 0.5, mid: float = 1.0, right: str = "call") -> LegQuote:
    return LegQuote(
        expiration=ASOF + pd.Timedelta(days=38),
        strike=strike,
        right=right,
        bid=mid - 0.05,
        ask=mid + 0.05,
        mid=mid,
        iv=0.30,
        delta=delta,
        dte=38,
    )


def test_gate_drops_junk_rows(make_chain) -> None:
    keys = [("call", 38, s) for s in (90.0, 95.0, 100.0, 105.0, 110.0, 115.0)]
    chain = make_chain(
        mids={k: 2.0 for k in keys},
        overrides={
            ("call", 38, 90.0): {"bid": 0.0},  # no bid
            ("call", 38, 95.0): {"bid": 1.0, "ask": 1.2},  # 18% of mid spread
            ("call", 38, 100.0): {"open_interest": 50.0},  # thin OI
            ("call", 38, 105.0): {"iv": float("nan")},  # junk IV
            ("call", 38, 115.0): {"iv": 0.0},  # zero IV
        },
    )

    quotes = liquid_quotes(
        chain, right="call", expiration=chain["expiration"].iloc[0], spot=100.0, asof=ASOF
    )

    assert [q.strike for q in quotes] == [110.0]
    q = quotes[0]
    assert q.mid == pytest.approx(2.0)
    assert q.dte == 38
    assert 0.0 < q.delta < 1.0  # BS delta computed from the chain IV


def test_gate_requires_min_open_interest_not_nan(make_chain) -> None:
    chain = make_chain(
        mids={("put", 38, 95.0): 2.0},
        overrides={("put", 38, 95.0): {"open_interest": float("nan")}},
    )
    quotes = liquid_quotes(
        chain, right="put", expiration=chain["expiration"].iloc[0], spot=100.0, asof=ASOF
    )
    assert quotes == []


def test_pick_expiration_nearest_window_midpoint(make_chain) -> None:
    chain = make_chain(
        mids={("call", 32, 100.0): 2.0, ("call", 44, 100.0): 2.5, ("call", 75, 100.0): 4.0}
    )

    exp = pick_expiration(chain, dte_lo=30, dte_hi=45, asof=ASOF)

    # midpoint of (30, 45) is 37.5: |32-37.5| = 5.5 beats |44-37.5| = 6.5; 75 is out of window
    assert exp == ASOF + pd.Timedelta(days=32)


def test_pick_expiration_none_outside_window(make_chain) -> None:
    chain = make_chain(mids={("call", 10, 100.0): 1.0, ("call", 75, 100.0): 4.0})
    assert pick_expiration(chain, dte_lo=30, dte_hi=45, asof=ASOF) is None
    assert pick_expiration(chain.iloc[0:0], dte_lo=30, dte_hi=45, asof=ASOF) is None


def test_strike_pickers() -> None:
    quotes = [
        _q(90.0, delta=0.8),
        _q(95.0, delta=0.65),
        _q(100.0, delta=0.5),
        _q(105.0, delta=0.35),
    ]

    assert by_delta(quotes, 0.65).strike == 95.0
    assert by_delta([], 0.65) is None
    assert at_or_below(quotes, 96.0).strike == 95.0
    assert at_or_below(quotes, 95.0).strike == 95.0
    assert at_or_below(quotes, 80.0) is None
    assert strictly_below(quotes, 95.0).strike == 90.0
    assert strictly_above(quotes, 100.0).strike == 105.0
    assert strictly_above(quotes, 105.0) is None
    assert nearest_strike(quotes, 101.0).strike == 100.0
    assert nearest_strike([], 101.0) is None

    # Put deltas are NEGATIVE (BS convention); negative target picks the right strike.
    put_quotes = [
        _q(95.0, delta=-0.30, right="put"),
        _q(90.0, delta=-0.65, right="put"),
    ]
    assert by_delta(put_quotes, -0.65).delta == -0.65


def test_liquid_quotes_tz_aware_asof_matches_naive(make_chain) -> None:
    chain = make_chain(mids={("call", 38, 110.0): 2.0})
    exp = chain["expiration"].iloc[0]
    naive_quotes = liquid_quotes(chain, right="call", expiration=exp, spot=100.0, asof=ASOF)
    tz_quotes = liquid_quotes(
        chain, right="call", expiration=exp, spot=100.0, asof=pd.Timestamp("2026-06-10", tz="UTC")
    )
    assert [q.strike for q in tz_quotes] == [q.strike for q in naive_quotes]


def test_atm_iv_uses_income_window_quote_nearest_spot(make_chain) -> None:
    chain = make_chain(
        mids={("call", 38, 100.0): 2.0, ("put", 38, 95.0): 1.5, ("call", 75, 100.0): 4.0},
        iv=0.42,
    )
    assert atm_iv(chain, spot=100.0, asof=ASOF) == pytest.approx(0.42)
    empty = chain.iloc[0:0]
    assert atm_iv(empty, spot=100.0, asof=ASOF) is None
