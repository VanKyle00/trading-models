"""Structure builders: payoff math, PoP vs hand-computed lognormal, gate fallbacks."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy.stats import norm

from tradinglib.strategist.structures import (
    build_structures,
    pop_market_implied,
    stock_plan,
)
from tradinglib.tournament.levels import Levels

ASOF = pd.Timestamp("2026-06-10")
LONG_LEVELS = Levels(entry=100.0, entry_type="market", stop=96.0, target=108.0, condition="t")
SHORT_LEVELS = Levels(entry=100.0, entry_type="market", stop=104.0, target=92.0, condition="t")

# (right, dte, strike) -> mid. Engineered so: 75-DTE call delta~0.65 -> K95;
# CSP/bull-put strikes sit under the 96 stop with credit >= width/3.
LONG_MIDS = {
    ("call", 75, 95.0): 8.0,
    ("call", 75, 100.0): 5.5,
    ("call", 75, 105.0): 3.6,
    ("call", 75, 110.0): 2.4,
    ("put", 38, 85.0): 0.6,
    ("put", 38, 90.0): 1.0,
    ("put", 38, 95.0): 2.8,
}
SHORT_MIDS = {
    ("put", 75, 90.0): 1.8,
    ("put", 75, 95.0): 2.6,
    ("put", 75, 105.0): 7.5,
    ("put", 75, 110.0): 11.0,
    ("call", 38, 105.0): 2.4,
    ("call", 38, 110.0): 0.7,
}


def test_pop_market_implied_matches_hand_computed_lognormal() -> None:
    # P(S_T < 110 | S0=100, vol=0.2, T=0.25), zero-rate lognormal:
    d = (math.log(110.0 / 100.0) + 0.5 * 0.2**2 * 0.25) / (0.2 * math.sqrt(0.25))
    p_below = float(norm.cdf(d))

    above = pop_market_implied(spot=100.0, level=110.0, vol=0.2, t_years=0.25, profit_above=True)
    below = pop_market_implied(spot=100.0, level=110.0, vol=0.2, t_years=0.25, profit_above=False)

    assert below == pytest.approx(p_below)
    assert above == pytest.approx(1.0 - p_below)
    assert below == pytest.approx(0.842, abs=1e-3)  # sanity anchor


def test_pop_market_implied_nonpositive_level_is_certain() -> None:
    # junk quotes can push a breakeven to or below zero; lognormal support is (0, inf)
    pop = pop_market_implied(spot=100.0, level=0.0, vol=0.2, t_years=0.25, profit_above=True)
    assert pop == 1.0
    pop = pop_market_implied(spot=100.0, level=-0.1, vol=0.2, t_years=0.25, profit_above=False)
    assert pop == 0.0


def test_stock_plan_long() -> None:
    s = stock_plan(LONG_LEVELS, "long")

    assert s.kind == "stock" and s.unit == "share" and s.legs == []
    assert s.max_loss == pytest.approx(4.0)  # entry-to-stop plan risk
    assert s.max_gain == pytest.approx(8.0)
    assert s.breakeven == pytest.approx(100.0)
    assert s.rr == pytest.approx(2.0)
    assert s.pop_market_implied is None and s.premium_yield is None
    assert s.warnings == []


def test_stock_plan_short_carries_borrow_warning() -> None:
    s = stock_plan(SHORT_LEVELS, "short")

    assert s.kind == "stock_short"
    assert s.max_loss == pytest.approx(4.0) and s.max_gain == pytest.approx(8.0)
    assert any("borrow" in w for w in s.warnings)


def test_long_call_when_spread_does_not_cut_cost(make_chain) -> None:
    from tradinglib.strategist.structures import long_option

    chain = make_chain(mids=LONG_MIDS)  # short leg at 110 saves only 2.4/8.0 = 30% < 35%
    s = long_option(chain, LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")

    assert s is not None and s.kind == "long_call" and s.unit == "contract"
    assert [leg["action"] for leg in s.legs] == ["buy"]
    assert s.legs[0]["strike"] == 95.0  # delta ~0.67 is nearest the 0.65 target
    assert s.legs[0]["dte"] == 75  # the only expiry inside 60-90
    assert s.premium == pytest.approx(8.0)
    assert s.max_loss == pytest.approx(8.0)
    assert s.max_gain is None and s.rr is None  # uncapped payoff
    assert s.breakeven == pytest.approx(103.0)
    expected_pop = pop_market_implied(
        spot=100.0, level=103.0, vol=0.30, t_years=75 / 365, profit_above=True
    )
    assert s.pop_market_implied == pytest.approx(expected_pop)


def test_call_debit_spread_when_short_leg_at_target_cuts_cost(make_chain) -> None:
    from tradinglib.strategist.structures import long_option

    mids = {**LONG_MIDS, ("call", 75, 110.0): 3.0}  # saves 3.0/8.0 = 37.5% > 35%
    s = long_option(make_chain(mids=mids), LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")

    assert s is not None and s.kind == "call_debit_spread"
    assert [(leg["action"], leg["strike"]) for leg in s.legs] == [("buy", 95.0), ("sell", 110.0)]
    assert s.premium == pytest.approx(5.0)
    assert s.max_loss == pytest.approx(5.0)
    assert s.max_gain == pytest.approx(10.0)  # width 15 - debit 5
    assert s.breakeven == pytest.approx(100.0)
    assert s.rr == pytest.approx(2.0)


def test_long_put_mirrors_for_short_stance(make_chain) -> None:
    from tradinglib.strategist.structures import long_option

    chain = make_chain(mids=SHORT_MIDS)  # spread saves 1.8/7.5 = 24% -> plain put
    s = long_option(chain, SHORT_LEVELS, spot=100.0, asof=ASOF, stance="short")

    assert s is not None and s.kind == "long_put"
    assert s.legs[0]["strike"] == 105.0  # delta ~-0.61 nearest -0.65
    assert s.breakeven == pytest.approx(97.5)  # 105 - 7.5
    expected_pop = pop_market_implied(
        spot=100.0, level=97.5, vol=0.30, t_years=75 / 365, profit_above=False
    )
    assert s.pop_market_implied == pytest.approx(expected_pop)


def test_put_debit_spread_for_short_stance(make_chain) -> None:
    from tradinglib.strategist.structures import long_option

    mids = {**SHORT_MIDS, ("put", 75, 90.0): 3.0}  # saves 3.0/7.5 = 40% > 35%
    s = long_option(make_chain(mids=mids), SHORT_LEVELS, spot=100.0, asof=ASOF, stance="short")

    assert s is not None and s.kind == "put_debit_spread"
    assert [(leg["action"], leg["strike"]) for leg in s.legs] == [("buy", 105.0), ("sell", 90.0)]
    assert s.premium == pytest.approx(4.5)
    assert s.max_gain == pytest.approx(10.5)  # width 15 - debit 4.5
    assert s.breakeven == pytest.approx(100.5)


def test_long_option_not_offered_without_directional_expiry(make_chain) -> None:
    from tradinglib.strategist.structures import long_option

    chain = make_chain(mids={("call", 38, 100.0): 2.0})  # income window only
    assert long_option(chain, LONG_LEVELS, spot=100.0, asof=ASOF, stance="long") is None


def test_cash_secured_put_strike_at_or_below_stop(make_chain) -> None:
    from tradinglib.strategist.structures import cash_secured_put

    s = cash_secured_put(make_chain(mids=LONG_MIDS), LONG_LEVELS, spot=100.0, asof=ASOF)

    assert s is not None and s.kind == "csp"
    assert s.legs[0]["strike"] == 95.0 and s.legs[0]["action"] == "sell"
    assert s.premium == pytest.approx(-2.8)  # credit received
    assert s.max_loss == pytest.approx(92.2)  # strike - credit, stock to zero
    assert s.max_gain == pytest.approx(2.8)
    assert s.breakeven == pytest.approx(92.2)
    assert s.premium_yield == pytest.approx(2.8 / 95.0)
    # scenario P/L with the stock AT the ticket stop (96): put expires OTM -> keep credit
    assert s.loss_at_stop == pytest.approx(-2.8)
    assert any("assignment" in w for w in s.warnings)


def test_bull_put_spread_credit_rule(make_chain) -> None:
    from tradinglib.strategist.structures import credit_spread

    s = credit_spread(make_chain(mids=LONG_MIDS), LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")

    assert s is not None and s.kind == "bull_put_spread"
    assert [(leg["action"], leg["strike"]) for leg in s.legs] == [("sell", 95.0), ("buy", 90.0)]
    assert s.premium == pytest.approx(-1.8)
    assert s.max_loss == pytest.approx(3.2)  # width 5 - credit 1.8
    assert s.max_gain == pytest.approx(1.8)
    assert s.breakeven == pytest.approx(93.2)
    assert s.premium_yield == pytest.approx(1.8 / 5.0)
    assert s.rr == pytest.approx(1.8 / 3.2)


def test_bull_put_spread_rejected_when_credit_below_third_of_width(make_chain) -> None:
    from tradinglib.strategist.structures import credit_spread

    mids = {**LONG_MIDS, ("put", 38, 95.0): 2.0, ("put", 38, 90.0): 1.0}  # credit 1.0 < 5/3
    s = credit_spread(make_chain(mids=mids), LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")

    assert s is None


def test_bear_call_spread_short_strike_above_invalidation(make_chain) -> None:
    from tradinglib.strategist.structures import credit_spread

    s = credit_spread(
        make_chain(mids=SHORT_MIDS), SHORT_LEVELS, spot=100.0, asof=ASOF, stance="short"
    )

    assert s is not None and s.kind == "bear_call_spread"
    assert [(leg["action"], leg["strike"]) for leg in s.legs] == [("sell", 105.0), ("buy", 110.0)]
    assert s.premium == pytest.approx(-1.7)
    assert s.max_loss == pytest.approx(3.3)
    assert s.breakeven == pytest.approx(106.7)
    expected_pop = pop_market_implied(
        spot=100.0, level=106.7, vol=0.30, t_years=38 / 365, profit_above=False
    )
    assert s.pop_market_implied == pytest.approx(expected_pop)


def test_build_structures_long_offers_everything_that_passes(make_chain) -> None:
    out = build_structures(make_chain(mids=LONG_MIDS), LONG_LEVELS, "long", spot=100.0, asof=ASOF)
    assert [s.kind for s in out] == ["stock", "long_call", "csp", "bull_put_spread"]


def test_build_structures_short_never_builds_naked_calls(make_chain) -> None:
    out = build_structures(
        make_chain(mids=SHORT_MIDS), SHORT_LEVELS, "short", spot=100.0, asof=ASOF
    )
    kinds = [s.kind for s in out]
    assert kinds == ["stock_short", "long_put", "bear_call_spread"]
    # naked short calls are never offered, by construction: no builder emits one
    assert not any(
        leg["action"] == "sell" and leg["right"] == "call" and len(s.legs) == 1
        for s in out
        for leg in s.legs
    )


def test_build_structures_empty_chain_is_stock_only(make_chain) -> None:
    empty = make_chain(mids={("call", 38, 100.0): 1.0}).iloc[0:0]
    out = build_structures(empty, LONG_LEVELS, "long", spot=100.0, asof=ASOF)
    assert [s.kind for s in out] == ["stock"]


def test_pop_range_market_implied_is_difference_of_tails() -> None:
    from tradinglib.strategist.structures import pop_range_market_implied

    below_hi = pop_market_implied(
        spot=100.0, level=112.0, vol=0.3, t_years=38 / 365, profit_above=False
    )
    below_lo = pop_market_implied(
        spot=100.0, level=88.0, vol=0.3, t_years=38 / 365, profit_above=False
    )

    pop = pop_range_market_implied(
        spot=100.0, lower=88.0, upper=112.0, vol_lower=0.3, vol_upper=0.3, t_years=38 / 365
    )

    assert pop == pytest.approx(below_hi - below_lo)
    assert 0.0 < pop < 1.0


def test_pop_range_market_implied_clamps_at_zero() -> None:
    from tradinglib.strategist.structures import pop_range_market_implied

    # an inverted band (junk breakevens) must clamp, not go negative
    pop = pop_range_market_implied(
        spot=100.0, lower=112.0, upper=88.0, vol_lower=0.3, vol_upper=0.3, t_years=38 / 365
    )
    assert pop == 0.0


def test_band_holds_range_levels() -> None:
    from tradinglib.strategist.structures import Band

    band = Band(lower=92.0, upper=108.0, condition="t")
    assert band.lower == 92.0 and band.upper == 108.0


# Neutral-band fixture: band 92-108 on spot 100, income window (38 DTE).
# Condor: shorts 90P/110C (strictly beyond the band), wings 85P/115C ($5 out),
# credit (1.6-0.6)+(1.5-0.5)=2.0 >= width 5 / 3. Butterfly: body 100, wings
# 90P/110C (nearest the band edges), credit 3.5+3.4-1.6-1.5=3.8, width 10.
NEUTRAL_MIDS = {
    ("put", 38, 85.0): 0.6,
    ("put", 38, 90.0): 1.6,
    ("put", 38, 95.0): 2.8,
    ("put", 38, 100.0): 3.4,
    ("call", 38, 100.0): 3.5,
    ("call", 38, 105.0): 2.4,
    ("call", 38, 110.0): 1.5,
    ("call", 38, 115.0): 0.5,
}


def _band():
    from tradinglib.strategist.structures import Band

    return Band(lower=92.0, upper=108.0, condition="t")


def test_iron_condor_shorts_beyond_band_wings_5_wide(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor, pop_range_market_implied

    s = iron_condor(make_chain(mids=NEUTRAL_MIDS), _band(), spot=100.0, asof=ASOF)

    assert s is not None and s.kind == "iron_condor" and s.unit == "contract"
    assert [(leg["action"], leg["right"], leg["strike"]) for leg in s.legs] == [
        ("buy", "put", 85.0),
        ("sell", "put", 90.0),
        ("sell", "call", 110.0),
        ("buy", "call", 115.0),
    ]
    assert s.premium == pytest.approx(-2.0)  # credit received
    assert s.max_loss == pytest.approx(3.0)  # widest wing 5 - credit 2
    assert s.max_gain == pytest.approx(2.0)
    assert s.breakeven is None
    assert s.breakevens == pytest.approx([88.0, 112.0])  # 90 - 2, 110 + 2
    assert s.rr == pytest.approx(2.0 / 3.0)
    assert s.premium_yield == pytest.approx(2.0 / 5.0)
    expected_pop = pop_range_market_implied(
        spot=100.0, lower=88.0, upper=112.0, vol_lower=0.30, vol_upper=0.30, t_years=38 / 365
    )
    assert s.pop_market_implied == pytest.approx(expected_pop)


def test_iron_condor_rejected_when_credit_below_third_of_width(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor

    # credit (0.8-0.6)+(0.8-0.5)=0.5 < 5/3
    mids = {**NEUTRAL_MIDS, ("put", 38, 90.0): 0.8, ("call", 38, 110.0): 0.8}
    assert iron_condor(make_chain(mids=mids), _band(), spot=100.0, asof=ASOF) is None


def test_iron_condor_not_offered_without_income_expiry(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor

    mids = {(right, 75, k): v for (right, _, k), v in NEUTRAL_MIDS.items()}  # 75 DTE only
    assert iron_condor(make_chain(mids=mids), _band(), spot=100.0, asof=ASOF) is None


def test_iron_condor_none_when_no_strike_beyond_band(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor

    mids = {k: v for k, v in NEUTRAL_MIDS.items() if k[2] not in (85.0, 90.0)}  # no put < 92
    assert iron_condor(make_chain(mids=mids), _band(), spot=100.0, asof=ASOF) is None


def test_iron_butterfly_body_at_spot_wings_at_band(make_chain) -> None:
    from tradinglib.strategist.structures import iron_butterfly

    s = iron_butterfly(make_chain(mids=NEUTRAL_MIDS), _band(), spot=100.0, asof=ASOF)

    assert s is not None and s.kind == "iron_butterfly" and s.unit == "contract"
    assert [(leg["action"], leg["right"], leg["strike"]) for leg in s.legs] == [
        ("buy", "put", 90.0),
        ("sell", "put", 100.0),
        ("sell", "call", 100.0),
        ("buy", "call", 110.0),
    ]
    assert s.premium == pytest.approx(-3.8)  # 3.4 + 3.5 - 1.6 - 1.5
    assert s.max_loss == pytest.approx(6.2)  # widest wing 10 - credit 3.8
    assert s.max_gain == pytest.approx(3.8)
    assert s.breakevens == pytest.approx([96.2, 103.8])  # body 100 -/+ credit


def test_iron_butterfly_none_without_matching_put_at_body(make_chain) -> None:
    from tradinglib.strategist.structures import iron_butterfly

    mids = {k: v for k, v in NEUTRAL_MIDS.items() if k != ("put", 38, 100.0)}
    assert iron_butterfly(make_chain(mids=mids), _band(), spot=100.0, asof=ASOF) is None


def test_build_neutral_structures_condor_first(make_chain) -> None:
    from tradinglib.strategist.structures import build_neutral_structures

    out = build_neutral_structures(make_chain(mids=NEUTRAL_MIDS), _band(), spot=100.0, asof=ASOF)
    assert [s.kind for s in out] == ["iron_condor", "iron_butterfly"]


def test_build_neutral_structures_empty_chain_is_empty(make_chain) -> None:
    from tradinglib.strategist.structures import build_neutral_structures

    empty = make_chain(mids={("call", 38, 100.0): 1.0}).iloc[0:0]
    assert build_neutral_structures(empty, _band(), spot=100.0, asof=ASOF) == []


def test_structure_theta_week_signs_and_magnitude(make_chain) -> None:
    from tradinglib.options.pricing import bs_greeks
    from tradinglib.strategist.structures import (
        credit_spread,
        long_option,
        structure_theta_week,
    )

    chain = make_chain(mids=LONG_MIDS)
    lc = long_option(chain, LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")
    assert lc is not None and lc.kind == "long_call"
    # single bought leg: K95 call, 75 DTE, iv 0.30 (conftest default)
    expected = bs_greeks("call", 100.0, 95.0, 75 / 365, 0.30, 0.0).theta * 7.0 / 365.0
    assert structure_theta_week(lc, spot=100.0) == pytest.approx(expected)
    assert structure_theta_week(lc, spot=100.0) < 0  # long premium decays

    cs = credit_spread(chain, LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")
    assert cs is not None
    assert structure_theta_week(cs, spot=100.0) > 0  # net short premium collects decay


def test_structure_theta_week_none_without_legs() -> None:
    from tradinglib.strategist.structures import structure_theta_week

    s = stock_plan(LONG_LEVELS, "long")
    assert structure_theta_week(s, spot=100.0) is None


# With iv=0.30, spot=100, 75 DTE: deltas are 90→0.80, 95→0.67, 100→0.53, 105→0.39, 110→0.26.
LADDER_MIDS = {**LONG_MIDS, ("call", 75, 90.0): 11.5}


def test_long_option_candidates_full_ladder_keys_and_strikes(make_chain) -> None:
    from tradinglib.strategist.structures import long_option_candidates

    out, notes = long_option_candidates(
        make_chain(mids=LADDER_MIDS), LONG_LEVELS, spot=100.0, asof=ASOF, stance="long"
    )
    by_key = {s.key: s for s in out}
    assert by_key["long_call_d65"].legs[0]["strike"] == 95.0
    assert by_key["long_call_d50"].legs[0]["strike"] == 100.0
    assert by_key["long_call_d80"].legs[0]["strike"] == 90.0
    assert out[0].key == "long_call_d65"  # anchor leads when no spread builds
    assert notes == []
    assert all(s.kind in ("long_call", "call_debit_spread") for s in out)


def test_long_option_candidates_dedupes_same_strike(make_chain) -> None:
    from tradinglib.strategist.structures import long_option_candidates

    # without a 90 strike the d80 pick lands on 95 — same as d65 — and dedupes silently
    out, notes = long_option_candidates(
        make_chain(mids=LONG_MIDS), LONG_LEVELS, spot=100.0, asof=ASOF, stance="long"
    )
    keys = [s.key for s in out]
    assert "long_call_d80" not in keys and "long_call_d65" in keys
    strikes = [s.legs[0]["strike"] for s in out if s.kind == "long_call"]
    assert len(strikes) == len(set(strikes))
    assert notes == []  # dedupe is not a liquidity drop


def test_long_option_candidates_spread_first_when_it_cuts_cost(make_chain) -> None:
    from tradinglib.strategist.structures import long_option_candidates

    mids = {**LADDER_MIDS, ("call", 75, 110.0): 3.0}  # saves 3.0/8.0 = 37.5% > 35%
    out, _ = long_option_candidates(
        make_chain(mids=mids), LONG_LEVELS, spot=100.0, asof=ASOF, stance="long"
    )
    assert out[0].kind == "call_debit_spread"  # family representative stays the spread
    assert out[0].key == "call_debit_spread"
    assert [(leg["action"], leg["strike"]) for leg in out[0].legs] == [
        ("buy", 95.0),
        ("sell", 110.0),
    ]
    assert "long_call_d65" in [s.key for s in out]  # plain anchor still a row


def test_long_option_candidates_empty_without_expiry(make_chain) -> None:
    from tradinglib.strategist.structures import long_option_candidates

    chain = make_chain(mids={("call", 38, 100.0): 2.0})  # income window only
    out, notes = long_option_candidates(chain, LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")
    assert out == [] and notes == []


def test_build_chat_structures_includes_income_rows_and_keys(make_chain) -> None:
    from tradinglib.strategist.structures import build_chat_structures

    out, notes = build_chat_structures(
        make_chain(mids=LADDER_MIDS), LONG_LEVELS, "long", spot=100.0, asof=ASOF
    )
    keys = [s.key for s in out]
    assert "csp" in keys and "bull_put_spread" in keys
    assert "stock" not in [s.kind for s in out]
    assert notes == []  # every ladder delta is liquid in this fixture


# Wide-condor fixture: regular shorts 90P/110C (credit (2.6-1.2)+(2.0-1.1)=2.3 >= 5/3),
# wide shorts one strike further at 85P/115C, wings 80P/120C
# (credit (1.2-0.3)+(1.1-0.2)=1.8 >= 5/3). Mids monotone in strike on both sides.
WIDE_MIDS = {
    ("put", 38, 80.0): 0.3,
    ("put", 38, 85.0): 1.2,
    ("put", 38, 90.0): 2.6,
    ("put", 38, 95.0): 2.8,
    ("put", 38, 100.0): 3.4,
    ("call", 38, 100.0): 3.5,
    ("call", 38, 105.0): 2.4,
    ("call", 38, 110.0): 2.0,
    ("call", 38, 115.0): 1.1,
    ("call", 38, 120.0): 0.2,
}


def test_iron_condor_widen_moves_both_shorts_one_strike_out(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor

    s = iron_condor(make_chain(mids=WIDE_MIDS), _band(), spot=100.0, asof=ASOF, widen=True)

    assert s is not None and s.kind == "iron_condor" and s.key == "condor_wide"
    assert [(leg["action"], leg["right"], leg["strike"]) for leg in s.legs] == [
        ("buy", "put", 80.0),
        ("sell", "put", 85.0),
        ("sell", "call", 115.0),
        ("buy", "call", 120.0),
    ]
    assert s.premium == pytest.approx(-1.8)
    assert s.max_loss == pytest.approx(3.2)  # widest wing 5 - credit 1.8
    assert s.breakevens == pytest.approx([83.2, 116.8])
    assert "wide" in s.label


def test_iron_condor_widen_none_without_further_short_strike(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor

    # no put below the regular 90 short: the widen step itself has nowhere to go
    mids = {k: v for k, v in NEUTRAL_MIDS.items() if k != ("put", 38, 85.0)}
    assert iron_condor(make_chain(mids=mids), _band(), spot=100.0, asof=ASOF, widen=True) is None


def test_iron_condor_widen_none_without_wing_beyond_wide_shorts(make_chain) -> None:
    from tradinglib.strategist.structures import iron_condor

    # NEUTRAL_MIDS: widened shorts land on 85P/115C but no strikes remain for the wings
    assert (
        iron_condor(make_chain(mids=NEUTRAL_MIDS), _band(), spot=100.0, asof=ASOF, widen=True)
        is None
    )


def test_build_neutral_structures_three_candidates(make_chain) -> None:
    from tradinglib.strategist.structures import build_neutral_structures

    out = build_neutral_structures(make_chain(mids=WIDE_MIDS), _band(), spot=100.0, asof=ASOF)
    assert [s.key for s in out] == ["condor", "condor_wide", "butterfly"]


def test_long_option_candidates_single_note_when_expiry_illiquid(make_chain) -> None:
    from tradinglib.strategist.structures import long_option_candidates

    # expiry listed in the 60-90 window but every quote fails the OI gate
    chain = make_chain(mids=LONG_MIDS, open_interest=50.0)
    out, notes = long_option_candidates(chain, LONG_LEVELS, spot=100.0, asof=ASOF, stance="long")

    assert out == []
    assert notes == ["no liquid call quotes in the 60-90 DTE window; long-option ladder dropped"]


def test_build_chat_structures_short_stance_mirrors(make_chain) -> None:
    from tradinglib.strategist.structures import build_chat_structures

    out, notes = build_chat_structures(
        make_chain(mids=SHORT_MIDS), SHORT_LEVELS, "short", spot=100.0, asof=ASOF
    )
    keys = [s.key for s in out]
    assert "csp" not in keys  # CSP is long-stance only
    assert "bear_call_spread" in keys
    assert any(k.startswith("long_put_d") for k in keys)
    assert notes == []
