"""Playbook against a REAL frozen yfinance chain (AAPL) — per the spec's fixture
rule: messy captured reality, never invented clean rows.

This snapshot was captured PRE-MARKET, where yfinance zeroes bids chain-wide, so
it pins the degradation path end-to-end on real data: the liquidity gate rejects
every leg and ``build_ticket`` falls back to a stock-only ticket that still
frames itself honestly. Builder math on quotable chains is covered by the
synthetic-chain tests; a market-hours recapture can extend this file to assert
option structures build on real quotes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglib.strategist import build_ticket
from tradinglib.strategist.quotes import liquid_quotes, pick_expiration

FIXTURE = Path(__file__).parent / "fixtures" / "options" / "aapl_chain_2026-06-10.parquet"


def _chain() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def _bars(spot: float) -> pd.DataFrame:
    close = spot * (1.0 + 0.01 * np.sin(np.arange(300) / 5.0))
    idx = pd.date_range("2025-06-01", periods=300, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=idx,
    )


def _winner(spot: float, stop_mult: float, target_mult: float) -> dict:
    return {
        "strategy": "sma_cross",
        "survived": True,
        "reasons": [],
        "params": {"fast": 10, "slow": 50},
        "deflated_sharpe": 0.93,
        "sharpe": 1.1,
        "n_trades": 18,
        "max_param_change_rate": 0.0,
        "n_windows": 6,
        "levels": {
            "entry": spot,
            "entry_type": "market",
            "stop": spot * stop_mult,
            "target": spot * target_mult,
            "condition": "fixture",
        },
    }


def test_gate_rejects_real_junk_rows_wholesale() -> None:
    chain = _chain()
    spot = float(chain["spot"].iloc[0])
    asof = pd.Timestamp(chain["date"].iloc[0])
    exp = pick_expiration(chain, dte_lo=30, dte_hi=45, asof=asof)
    assert exp is not None  # a monthly expiration sits in the income window

    quotes = liquid_quotes(chain, right="put", expiration=exp, spot=spot, asof=asof)
    raw = chain[(chain["right"] == "put") & (chain["expiration"] == exp)]

    assert len(raw) > 0
    assert len(quotes) < len(raw)  # the gate dropped junk rows
    # this capture is pre-market: yfinance zeroes bids chain-wide, nothing is quotable
    assert (chain["bid"] <= 0).mean() > 0.5
    assert quotes == []


def test_build_ticket_degrades_to_stock_only_on_unquotable_real_chain() -> None:
    chain = _chain()
    spot = float(chain["spot"].iloc[0])
    for stance, stop_mult, target_mult in (("long", 0.95, 1.10), ("short", 1.05, 0.90)):
        ticket = build_ticket(
            ticker="AAPL",
            stance=stance,
            winner=_winner(spot, stop_mult, target_mult),
            bars=_bars(spot),
            chain=chain,
        )

        kinds = [s["kind"] for s in ticket["structures"]]
        assert kinds == (["stock"] if stance == "long" else ["stock_short"])
        assert sum(s["recommended"] for s in ticket["structures"]) == 1
        assert any("options_illiquid" in w for w in ticket["warnings"])
        assert ticket["quotes_asof"] == "2026-06-10"  # a chain was present, just unquotable
        for s in ticket["structures"]:
            for field in ("max_loss", "max_gain", "breakeven"):
                if s[field] is not None:
                    assert np.isfinite(s[field]), (stance, s["kind"], field)
        json.dumps(ticket)  # report.json must round-trip the real thing
