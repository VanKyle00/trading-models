"""build_hypothesis_ticket: chat-path ticket — preference pinning, no evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.options.surface import realized_vol
from tradinglib.strategist import build_hypothesis_ticket

LEVELS = {
    "entry": 100.0,
    "entry_type": "market",
    "stop": 96.0,
    "target": 108.0,
    "condition": "user: bullish on TEST",
}

MIDS = {
    ("call", 75, 95.0): 8.0,
    ("call", 75, 100.0): 5.5,
    ("call", 75, 105.0): 3.6,
    ("call", 75, 110.0): 2.4,
    ("put", 38, 85.0): 0.6,
    ("put", 38, 90.0): 1.0,
    ("put", 38, 95.0): 2.8,
}


def _bars() -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(300) / 5.0)
    idx = pd.date_range("2025-06-01", periods=300, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=idx,
    )


def _rv() -> float:
    return float(realized_vol(_bars()["close"]).iloc[-1])


def _ticket(make_chain, *, preference: str = "auto", iv: float | None = None, **kwargs) -> dict:
    chain = make_chain(mids=MIDS, iv=iv if iv is not None else _rv())  # neutral tilt by default
    return build_hypothesis_ticket(
        ticker="TEST",
        stance="long",
        levels=dict(LEVELS),
        bars=_bars(),
        chain=chain,
        preference=preference,
        **kwargs,
    )


def test_auto_preference_neutral_iv_recommends_directional(make_chain) -> None:
    ticket = _ticket(make_chain)

    # at the fixture's low realized vol the 0.65-delta pick is the ATM call and
    # the short leg at the target cuts cost > 35%, so the directional structure
    # long_option builds is the 100/110 debit spread
    rec = [s for s in ticket["structures"] if s["recommended"]]
    assert len(rec) == 1 and rec[0]["kind"] == "call_debit_spread"
    assert rec[0]["quantity"] == 3  # 100k * 1% // ($3.10 debit x 100)


def test_auto_preference_high_iv_tilts_to_premium(make_chain) -> None:
    ticket = _ticket(make_chain, iv=1.5 * _rv())

    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "bull_put_spread"
    assert ticket["structures"][0]["kind"] == "bull_put_spread"  # ordering, not just the flag


def test_pinned_directional_ignores_high_iv(make_chain) -> None:
    ticket = _ticket(make_chain, preference="directional", iv=1.5 * _rv())

    assert ticket["structures"][0]["kind"] == "call_debit_spread"
    assert ticket["iv_ratio"] is not None  # still reported for framing, just not applied


def test_pinned_premium_ignores_low_iv(make_chain) -> None:
    ticket = _ticket(make_chain, preference="premium", iv=0.5 * _rv())

    assert ticket["structures"][0]["kind"] == "bull_put_spread"


def test_chat_framing_no_tournament_fields(make_chain) -> None:
    ticket = _ticket(make_chain)

    assert ticket["source"] == "chat"
    assert "strategy" not in ticket and "style" not in ticket
    assert "params" not in ticket and "evidence" not in ticket
    assert ticket["levels"]["condition"] == "user: bullish on TEST"
    assert ticket["account_size"] == 100_000.0
    assert ticket["risk_per_trade_pct"] == 0.01
    assert any("indicative" in w for w in ticket["warnings"])
    assert sum(s["recommended"] for s in ticket["structures"]) == 1


def test_earnings_demotes_undefined_risk(make_chain) -> None:
    ticket = _ticket(
        make_chain,
        preference="premium",
        earnings_warning=True,
        next_earnings=pd.Timestamp("2026-06-20", tz="UTC"),  # before every option expiry
    )

    kinds = [s["kind"] for s in ticket["structures"]]
    assert kinds.index("csp") > kinds.index("bull_put_spread")
    assert any("earnings" in w for w in ticket["warnings"])


def test_bad_preference_raises(make_chain) -> None:
    with pytest.raises(ValueError, match="preference"):
        _ticket(make_chain, preference="yolo")


def test_chat_ticket_never_contains_stock_plan(make_chain) -> None:
    ticket = _ticket(make_chain)

    kinds = [s["kind"] for s in ticket["structures"]]
    assert "stock" not in kinds and "stock_short" not in kinds
    assert all(s["legs"] for s in ticket["structures"])  # every structure is options


def test_chat_ticket_all_illiquid_raises(make_chain) -> None:
    empty = make_chain(mids={("call", 38, 100.0): 1.0}).iloc[0:0]

    with pytest.raises(ValueError, match="liquidity gate"):
        build_hypothesis_ticket(
            ticker="TEST",
            stance="long",
            levels=dict(LEVELS),
            bars=_bars(),
            chain=empty,
        )


NEUTRAL_LEVELS = {"lower": 92.0, "upper": 108.0, "condition": "user: TEST range-bound"}

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


def _neutral_ticket(make_chain, **kwargs) -> dict:
    return build_hypothesis_ticket(
        ticker="TEST",
        stance="neutral",
        levels=dict(NEUTRAL_LEVELS),
        bars=_bars(),
        chain=make_chain(mids=NEUTRAL_MIDS),
        **kwargs,
    )


def test_neutral_ticket_recommends_condor_first(make_chain) -> None:
    ticket = _neutral_ticket(make_chain)

    assert ticket["stance"] == "neutral"
    assert ticket["levels"] == NEUTRAL_LEVELS
    assert [s["kind"] for s in ticket["structures"]] == ["iron_condor", "iron_butterfly"]
    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "iron_condor"
    assert rec["quantity"] == 3  # 100k * 1% // ($3.00 max loss x 100)
    assert ticket["iv_ratio"] is not None  # framing still reported
    assert any("indicative" in w for w in ticket["warnings"])


def test_neutral_ticket_spanning_earnings_warns_per_structure(make_chain) -> None:
    ticket = _neutral_ticket(
        make_chain,
        earnings_warning=True,
        next_earnings=pd.Timestamp("2026-06-20", tz="UTC"),  # before the 38-DTE expiry
    )

    assert all(any("earnings" in w for w in s["warnings"]) for s in ticket["structures"])
    assert any("earnings" in w for w in ticket["warnings"])


def test_neutral_ticket_all_illiquid_raises(make_chain) -> None:
    empty = make_chain(mids={("call", 38, 100.0): 1.0}).iloc[0:0]

    with pytest.raises(ValueError, match="liquidity gate"):
        build_hypothesis_ticket(
            ticker="TEST",
            stance="neutral",
            levels=dict(NEUTRAL_LEVELS),
            bars=_bars(),
            chain=empty,
        )


def test_chat_ticket_has_candidate_ladder_with_keys(make_chain) -> None:
    ticket = _ticket(make_chain)

    keys = [s["key"] for s in ticket["structures"]]
    assert all(keys), "every chat structure carries a stable key"
    assert len(keys) == len(set(keys))
    assert sum(1 for k in keys if k.startswith("long_call_d")) >= 2  # the ladder
    assert all(s["quantity"] is not None for s in ticket["structures"])  # sizing ran on every row
    assert any(s["quantity"] for s in ticket["structures"])  # at least one row sized


def test_chat_ticket_recommended_has_reason_and_theta(make_chain) -> None:
    ticket = _ticket(make_chain)

    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["reason"]  # deterministic explanation, e.g. preference/tilt text
    others = [s for s in ticket["structures"] if not s["recommended"]]
    assert all(s["reason"] is None for s in others)
    assert all(s["theta_week"] is not None for s in ticket["structures"])


def test_chat_ticket_plan_present_and_for_recommended(make_chain) -> None:
    ticket = _ticket(make_chain)

    plan = ticket["plan"]
    assert plan is not None
    assert len(plan["price_rules"]) == 2
    assert plan["profit_take"] is not None and plan["loss_cut"] is not None
    assert plan["est_by"]
    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert plan["theta_week"] == rec["theta_week"]


def test_structure_key_pins_recommendation_and_recomputes_plan(make_chain) -> None:
    base = _ticket(make_chain)
    pin = next(s["key"] for s in base["structures"] if not s["recommended"])

    pinned = _ticket(make_chain, structure_key=pin)

    rec = next(s for s in pinned["structures"] if s["recommended"])
    assert rec["key"] == pin
    assert rec["reason"] == "pinned by user request"
    assert sum(s["recommended"] for s in pinned["structures"]) == 1
    assert pinned["plan"]["theta_week"] == rec["theta_week"]


def test_unknown_structure_key_lists_valid_keys(make_chain) -> None:
    with pytest.raises(ValueError, match="valid keys"):
        _ticket(make_chain, structure_key="nope")


# ---------------------------------------------------------------------------
# Fix 1: fall-through recommendation must not carry a contradictory tilt reason
# ---------------------------------------------------------------------------


def test_fallthrough_recommendation_reason_not_contradictory(make_chain) -> None:
    # every long-call row costs > the $1k risk budget (qty = 0); the bull_put_spread
    # sizes (credit 2.8-1.0=1.8 on width 5, max_loss 3.2, qty = 1000//320 = 3)
    pricey = {
        ("call", 75, 95.0): 18.0,
        ("call", 75, 100.0): 14.0,
        ("put", 38, 90.0): 1.0,
        ("put", 38, 95.0): 2.8,
    }
    chain = make_chain(mids=pricey, iv=0.5 * _rv())  # buy-premium tilt
    ticket = build_hypothesis_ticket(
        ticker="TEST", stance="long", levels=dict(LEVELS), bars=_bars(), chain=chain
    )

    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "bull_put_spread"  # long rows all unsized; spread sized
    assert "favors buying premium" not in rec["reason"]
    assert "first candidate to size" in rec["reason"]


# ---------------------------------------------------------------------------
# Fix 2: neutral ticket-level coverage
# ---------------------------------------------------------------------------

# mirrors tests/test_strategist_structures.py: regular + wide condor + butterfly all build
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

_NEUTRAL_LEVELS = {"lower": 92.0, "upper": 108.0, "condition": "user: TEST stays 92-108"}


def test_neutral_ticket_three_candidates_reason_and_band_plan(make_chain) -> None:
    chain = make_chain(mids=WIDE_MIDS, iv=_rv())
    ticket = build_hypothesis_ticket(
        ticker="TEST", stance="neutral", levels=dict(_NEUTRAL_LEVELS), bars=_bars(), chain=chain
    )

    assert [s["key"] for s in ticket["structures"]] == ["condor", "condor_wide", "butterfly"]
    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["key"] == "condor"
    assert rec["reason"] == "shorts just beyond your band — best credit-for-room balance"
    assert all(s["theta_week"] is not None for s in ticket["structures"])
    lo, hi = ticket["plan"]["price_rules"]
    assert "92" in lo["trigger"] and "band floor" in lo["trigger"]
    assert "108" in hi["trigger"] and "band ceiling" in hi["trigger"]


def test_neutral_structure_key_pins_butterfly(make_chain) -> None:
    chain = make_chain(mids=WIDE_MIDS, iv=_rv())
    base = build_hypothesis_ticket(
        ticker="TEST", stance="neutral", levels=dict(_NEUTRAL_LEVELS), bars=_bars(), chain=chain
    )
    pinned = build_hypothesis_ticket(
        ticker="TEST",
        stance="neutral",
        levels=dict(_NEUTRAL_LEVELS),
        bars=_bars(),
        chain=chain,
        structure_key="butterfly",
    )

    rec = next(s for s in pinned["structures"] if s["recommended"])
    assert rec["key"] == "butterfly" and rec["reason"] == "pinned by user request"
    # the plan really was recomputed for the pinned structure, not copied
    assert (
        pinned["plan"]["price_rules"][0]["est_value"] != base["plan"]["price_rules"][0]["est_value"]
    )


# ---------------------------------------------------------------------------
# Fix 3: empty-string pin should error, not silently no-op
# ---------------------------------------------------------------------------


def test_empty_structure_key_raises(make_chain) -> None:
    with pytest.raises(ValueError, match="valid keys"):
        _ticket(make_chain, structure_key="")


# ---------------------------------------------------------------------------
# Fix 4: strengthen the sizing assertion (in test_chat_ticket_has_candidate_ladder_with_keys)
# ---------------------------------------------------------------------------
# (The existing test is updated below; this section documents the change.)
