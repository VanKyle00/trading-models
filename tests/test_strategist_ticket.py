"""build_ticket: deterministic selection, IV tilt, earnings demotion, framing (C3/C5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.loaders.options.yf_chain import CHAIN_COLUMNS
from tradinglib.options.surface import realized_vol
from tradinglib.strategist import build_ticket

ASOF = "2026-06-10"
LEVELS = {"entry": 100.0, "entry_type": "market", "stop": 96.0, "target": 108.0, "condition": "t"}

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


def _winner(strategy: str = "sma_cross") -> dict:
    return {
        "strategy": strategy,
        "survived": True,
        "reasons": [],
        "params": {"fast": 10, "slow": 50},
        "deflated_sharpe": 0.93,
        "sharpe": 1.1,
        "n_trades": 18,
        "max_param_change_rate": 0.0,
        "n_windows": 6,
        "levels": dict(LEVELS),
    }


def _rv() -> float:
    return float(realized_vol(_bars()["close"]).iloc[-1])


def _ticket(make_chain, *, strategy: str = "sma_cross", iv: float | None = None, **kwargs) -> dict:
    chain = make_chain(mids=MIDS, iv=iv if iv is not None else _rv())  # neutral tilt by default
    return build_ticket(
        ticker="TEST",
        stance="long",
        winner=_winner(strategy),
        bars=_bars(),
        chain=chain,
        **kwargs,
    )


def test_trend_winner_recommends_directional_first(make_chain) -> None:
    ticket = _ticket(make_chain)  # sma_cross is style "trend"

    rec = [s for s in ticket["structures"] if s["recommended"]]
    assert len(rec) == 1 and rec[0]["kind"] == "stock"
    assert rec[0]["quantity"] == 250
    assert ticket["style"] == "trend"


def test_mean_reversion_winner_recommends_premium_first(make_chain) -> None:
    ticket = _ticket(make_chain, strategy="bollinger")

    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "bull_put_spread"
    assert ticket["structures"][0]["kind"] == "bull_put_spread"  # ordering, not just the flag


def test_high_iv_ratio_tilts_trend_winner_to_premium(make_chain) -> None:
    ticket = _ticket(make_chain, iv=1.5 * _rv())

    assert ticket["iv_ratio"] == pytest.approx(1.5, abs=0.05)
    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "bull_put_spread"


def test_low_iv_ratio_tilts_mean_reversion_winner_to_directional(make_chain) -> None:
    ticket = _ticket(make_chain, strategy="bollinger", iv=0.5 * _rv())

    rec = next(s for s in ticket["structures"] if s["recommended"])
    assert rec["kind"] == "stock"


def test_earnings_demotes_undefined_risk_below_defined(make_chain) -> None:
    ticket = _ticket(
        make_chain,
        strategy="bollinger",
        earnings_warning=True,
        next_earnings=pd.Timestamp("2026-06-20", tz="UTC"),  # before every option expiry
    )

    kinds = [s["kind"] for s in ticket["structures"]]
    assert kinds.index("csp") > kinds.index("bull_put_spread")
    assert kinds.index("stock") > kinds.index("bull_put_spread")
    csp = next(s for s in ticket["structures"] if s["kind"] == "csp")
    assert any("earnings" in w for w in csp["warnings"])
    assert any("earnings" in w for w in ticket["warnings"])


def test_empty_chain_degrades_to_stock_only_with_warning(make_chain) -> None:
    ticket = build_ticket(
        ticker="TEST",
        stance="long",
        winner=_winner(),
        bars=_bars(),
        chain=pd.DataFrame(columns=CHAIN_COLUMNS),
    )

    assert [s["kind"] for s in ticket["structures"]] == ["stock"]
    assert ticket["structures"][0]["recommended"] is True
    assert any("options_illiquid" in w for w in ticket["warnings"])
    assert ticket["quotes_asof"] is None and ticket["iv_ratio"] is None


def test_ticket_framing_and_evidence(make_chain) -> None:
    ticket = _ticket(
        make_chain,
        fa={"ticker": "TEST", "name": "T", "sector": "Tech", "fa_score": 0.91, "rank": 3},
        winner_changed=True,
    )

    assert ticket["ticker"] == "TEST" and ticket["stance"] == "long"
    assert ticket["strategy"] == "sma_cross" and ticket["params"] == {"fast": 10, "slow": 50}
    assert ticket["levels"] == LEVELS
    ev = ticket["evidence"]
    assert ev["deflated_sharpe"] == 0.93 and ev["sharpe"] == 1.1
    assert ev["n_trades"] == 18 and ev["n_windows"] == 6
    assert ev["winner_changed"] is True
    assert ev["fa_rank"] == 3 and ev["fa_score"] == 0.91
    assert ticket["quotes_asof"] == ASOF
    assert any("indicative" in w for w in ticket["warnings"])
    assert sum(s["recommended"] for s in ticket["structures"]) == 1
