"""planner: bar/chain/earnings plumbing for the options-planner chat tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tradinglib.assistant.planner as planner
from tradinglib.features.technical import atr


def _bars(n: int = 300) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(n) / 5.0)
    idx = pd.date_range("2025-06-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=idx,
    )


def test_propose_levels_long_uses_atr_stop_and_2r_target(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)

    out = planner.propose_levels("test", "long")

    entry = float(bars["close"].iloc[-1])
    atr14 = float(atr(bars["high"], bars["low"], bars["close"], 14).iloc[-1])
    assert out["ticker"] == "TEST"  # upper-cased
    assert out["levels"]["entry"] == pytest.approx(entry, abs=0.01)
    assert out["levels"]["entry_type"] == "market"
    assert out["levels"]["stop"] == pytest.approx(entry - 2 * atr14, abs=0.01)
    assert out["levels"]["target"] == pytest.approx(
        entry + 2 * (entry - out["levels"]["stop"]), abs=0.02
    )
    assert out["atr14"] == pytest.approx(atr14, abs=0.01)
    assert out["asof"] == bars.index[-1].strftime("%Y-%m-%d")


def test_propose_levels_short_mirrors(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)

    out = planner.propose_levels("TEST", "short")

    lv = out["levels"]
    assert lv["stop"] > lv["entry"] > lv["target"]


def test_propose_levels_empty_bars_raises(monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError, match="no daily bars"):
        planner.propose_levels("NOPE", "long")


MIDS = {
    ("call", 75, 95.0): 8.0,
    ("call", 75, 100.0): 5.5,
    ("call", 75, 105.0): 3.6,
    ("call", 75, 110.0): 2.4,
    ("put", 38, 85.0): 0.6,
    ("put", 38, 90.0): 1.0,
    ("put", 38, 95.0): 2.8,
}

LEVELS = {"entry": 100.0, "entry_type": "market", "stop": 96.0, "target": 108.0}


def _no_earnings(*a, **k) -> pd.DataFrame:
    return pd.DataFrame({"earnings_datetime": pd.DatetimeIndex([], tz="UTC")})


def test_hypothesis_ticket_end_to_end(make_chain, monkeypatch) -> None:
    from tradinglib.options.surface import realized_vol

    bars = _bars()
    rv = float(realized_vol(bars["close"]).iloc[-1])
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    monkeypatch.setattr(planner, "fetch_chain", lambda t: make_chain(mids=MIDS, iv=rv))
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    ticket = planner.hypothesis_ticket(
        ticker="test",
        stance="long",
        levels=dict(LEVELS),
        account_size=100_000.0,
        risk_per_trade_pct=0.01,
        hypothesis="bullish on TEST into summer",
    )

    assert ticket["source"] == "chat" and ticket["ticker"] == "TEST"
    assert ticket["levels"]["condition"] == "bullish on TEST into summer"
    assert ticket["next_earnings"] is None
    assert sum(s["recommended"] for s in ticket["structures"]) == 1
    assert all(s["legs"] for s in ticket["structures"])  # options only, no stock plan
    option = next(s for s in ticket["structures"] if s["legs"])
    assert option["calculator_url"].startswith("https://optionstrat.com/build/custom/TEST/")


def test_optionstrat_url_encodes_legs() -> None:
    structure = {
        "kind": "bull_put_spread",
        "quantity": 3,
        "legs": [
            {
                "action": "sell",
                "right": "put",
                "strike": 95.0,
                "expiration": "2026-07-18",
                "mid": 2.8,
            },
            {
                "action": "buy",
                "right": "put",
                "strike": 90.5,
                "expiration": "2026-07-18",
                "mid": 1.0,
            },
        ],
    }

    url = planner.optionstrat_url("TEST", structure)

    assert url == (
        "https://optionstrat.com/build/custom/TEST/-.TEST260718P95x3@2.80,.TEST260718P90.5x3@1.00"
    )


def test_optionstrat_url_unsized_defaults_to_one_contract() -> None:
    structure = {
        "kind": "long_call",
        "quantity": 0,  # unsized still gets a previewable link
        "legs": [
            {
                "action": "buy",
                "right": "call",
                "strike": 100.0,
                "expiration": "2026-08-24",
                "mid": 5.5,
            },
        ],
    }

    url = planner.optionstrat_url("TEST", structure)

    assert url == "https://optionstrat.com/build/custom/TEST/.TEST260824C100x1@5.50"


def test_optionstrat_url_stock_plan_is_none() -> None:
    assert planner.optionstrat_url("TEST", {"kind": "stock", "legs": []}) is None


def test_hypothesis_ticket_earnings_inside_warn_window_flags(make_chain, monkeypatch) -> None:
    # Asserts the warning + next_earnings framing. Demotion ORDER isn't asserted
    # here: the fixture chain's expirations fall before the earnings date, so only
    # the no-expiry stock plan spans it — kind-order tests live in Task 1's file.
    bars = _bars()
    soon = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=7)

    def _earnings(*a, **k) -> pd.DataFrame:
        return pd.DataFrame({"earnings_datetime": pd.DatetimeIndex([soon])})

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    monkeypatch.setattr(planner, "fetch_chain", lambda t: make_chain(mids=MIDS))
    monkeypatch.setattr(planner, "get_earnings_dates", _earnings)

    ticket = planner.hypothesis_ticket(
        ticker="TEST",
        stance="long",
        levels=dict(LEVELS),
        account_size=100_000.0,
        risk_per_trade_pct=0.01,
    )

    assert ticket["next_earnings"] == soon.tz_convert("UTC").tz_localize(None).strftime("%Y-%m-%d")
    assert any("earnings" in w for w in ticket["warnings"])


def test_hypothesis_ticket_earnings_failure_is_nonfatal(make_chain, monkeypatch) -> None:
    def _boom(*a, **k):
        raise ConnectionError("rate limited")

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", lambda t: make_chain(mids=MIDS))
    monkeypatch.setattr(planner, "get_earnings_dates", _boom)

    ticket = planner.hypothesis_ticket(
        ticker="TEST",
        stance="long",
        levels=dict(LEVELS),
        account_size=100_000.0,
        risk_per_trade_pct=0.01,
    )

    assert ticket["next_earnings"] is None  # lost the demotion signal, kept the ticket
    assert any("earnings lookup failed" in w for w in ticket["warnings"])


def test_hypothesis_ticket_chain_failure_propagates(monkeypatch) -> None:
    def _boom(t):
        raise ConnectionError("yfinance 429")

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", _boom)

    with pytest.raises(ConnectionError):
        planner.hypothesis_ticket(
            ticker="TEST",
            stance="long",
            levels=dict(LEVELS),
            account_size=100_000.0,
            risk_per_trade_pct=0.01,
        )
