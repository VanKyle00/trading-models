"""build_hypothesis_ticket: chat-path ticket — preference pinning, no evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    rec = [s for s in ticket["structures"] if s["recommended"]]
    assert len(rec) == 1 and rec[0]["kind"] == "stock"
    assert rec[0]["quantity"] == 250  # 100k * 1% // $4 risk per share


def test_auto_preference_high_iv_tilts_to_premium(make_chain) -> None:
    ticket = _ticket(make_chain, iv=1.5 * _rv())

    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "bull_put_spread"
    assert ticket["structures"][0]["kind"] == "bull_put_spread"  # ordering, not just the flag


def test_pinned_directional_ignores_high_iv(make_chain) -> None:
    ticket = _ticket(make_chain, preference="directional", iv=1.5 * _rv())

    assert ticket["structures"][0]["kind"] == "stock"
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
    assert kinds.index("stock") > kinds.index("bull_put_spread")
    assert any("earnings" in w for w in ticket["warnings"])


def test_bad_preference_raises(make_chain) -> None:
    import pytest

    with pytest.raises(ValueError, match="preference"):
        _ticket(make_chain, preference="yolo")
