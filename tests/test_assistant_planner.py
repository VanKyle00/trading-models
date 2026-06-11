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


def _frame(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", periods=len(close), freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx
    )


def _no_earnings(*a, **k) -> pd.DataFrame:
    return pd.DataFrame({"earnings_datetime": pd.DatetimeIndex([], tz="UTC"), "session": []})


def _quiet_events(monkeypatch) -> None:
    """No scheduled events — without the patches propose_levels would fetch live."""
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)
    monkeypatch.setattr(planner, "get_next_ex_dividend", lambda t: None)


def test_propose_levels_long_uses_atr_stop_and_2r_target(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

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
    _quiet_events(monkeypatch)

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
        return pd.DataFrame({"earnings_datetime": pd.DatetimeIndex([soon]), "session": ["amc"]})

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


def test_propose_levels_neutral_returns_atr_band(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("test", "neutral")

    spot = float(bars["close"].iloc[-1])
    atr14 = float(atr(bars["high"], bars["low"], bars["close"], 14).iloc[-1])
    assert out["ticker"] == "TEST" and out["stance"] == "neutral"
    assert out["levels"]["lower"] == pytest.approx(spot - 2 * atr14, abs=0.01)
    assert out["levels"]["upper"] == pytest.approx(spot + 2 * atr14, abs=0.01)
    assert "entry" not in out["levels"]
    assert "band" in out["method"]


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


def test_hypothesis_ticket_neutral_end_to_end(make_chain, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", lambda t: make_chain(mids=NEUTRAL_MIDS))
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    ticket = planner.hypothesis_ticket(
        ticker="test",
        stance="neutral",
        levels={"lower": 92.0, "upper": 108.0},
        account_size=100_000.0,
        risk_per_trade_pct=0.01,
        hypothesis="TEST stays range-bound into expiry",
    )

    assert ticket["ticker"] == "TEST" and ticket["stance"] == "neutral"
    assert ticket["levels"]["condition"] == "TEST stays range-bound into expiry"
    assert [s["kind"] for s in ticket["structures"]] == ["iron_condor", "iron_butterfly"]
    assert all(
        s["calculator_url"].startswith("https://optionstrat.com/build/custom/TEST/")
        for s in ticket["structures"]
    )


# ── guided scenarios, events, sparkline ────────────────────────────────


def test_propose_levels_offers_scenarios_one_recommended(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("TEST", "long")

    keys = [s["key"] for s in out["scenarios"]]
    assert keys[0] == "balanced" and "tight" in keys
    assert sum(s["recommended"] for s in out["scenarios"]) == 1
    assert out["levels"] == out["scenarios"][0]["levels"]  # top-level = recommended
    balanced = out["scenarios"][0]["levels"]
    tight = next(s for s in out["scenarios"] if s["key"] == "tight")["levels"]
    assert balanced["entry"] - tight["stop"] < balanced["entry"] - balanced["stop"]
    for s in out["scenarios"]:  # every offered scenario has valid long geometry
        lv = s["levels"]
        assert lv["stop"] < lv["entry"] < lv["target"]


def test_propose_levels_structure_scenario_anchors_on_swing(monkeypatch) -> None:
    n = 300
    close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    low[-10] = 95.0  # 20d swing low
    high[-5] = 106.0  # 20d swing high, > 1R away
    bars = _frame(close, high, low)
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("TEST", "long")

    atr14 = float(atr(bars["high"], bars["low"], bars["close"], 14).iloc[-1])
    struct = next(s for s in out["scenarios"] if s["key"] == "structure")
    assert struct["levels"]["stop"] == pytest.approx(95.0 - 0.25 * atr14, abs=0.01)
    assert struct["levels"]["target"] == pytest.approx(106.0, abs=0.01)
    assert not struct["recommended"]


def test_propose_levels_structure_scenario_gated_out_when_reward_thin(monkeypatch) -> None:
    # steady uptrend: spot sits at the 20d high, so the swing target offers < 1R
    n = 300
    close = np.linspace(80.0, 100.0, n)
    bars = _frame(close, close + 1.0, close - 1.0)
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("TEST", "long")

    assert all(s["key"] != "structure" for s in out["scenarios"])


def test_propose_levels_neutral_offers_range_when_it_brackets(monkeypatch) -> None:
    n = 300
    close = 100.0 + 5.0 * np.sin(np.arange(n) / 5.0)  # wide oscillation around spot
    bars = _frame(close, close + 1.0, close - 1.0)
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("TEST", "neutral")

    keys = [s["key"] for s in out["scenarios"]]
    assert keys[0] == "balanced" and "wide" in keys and "range" in keys
    assert sum(s["recommended"] for s in out["scenarios"]) == 1
    balanced = out["scenarios"][0]["levels"]
    wide = next(s for s in out["scenarios"] if s["key"] == "wide")["levels"]
    assert wide["lower"] < balanced["lower"] and wide["upper"] > balanced["upper"]
    rng = next(s for s in out["scenarios"] if s["key"] == "range")["levels"]
    assert rng["lower"] == pytest.approx(float(bars["low"].iloc[-20:].min()), abs=0.01)
    assert rng["upper"] == pytest.approx(float(bars["high"].iloc[-20:].max()), abs=0.01)


def test_propose_levels_neutral_range_gated_out_in_trend(monkeypatch) -> None:
    # uptrend: spot hugs the 20d high — no room above, range scenario withheld
    n = 300
    close = np.linspace(80.0, 100.0, n)
    bars = _frame(close, close + 1.0, close - 1.0)
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("TEST", "neutral")

    assert all(s["key"] != "range" for s in out["scenarios"])


def test_propose_levels_events_warn_inside_window(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    now = pd.Timestamp.now(tz="UTC")
    soon = now + pd.Timedelta(days=7)

    def _earnings(*a, **k) -> pd.DataFrame:
        return pd.DataFrame({"earnings_datetime": pd.DatetimeIndex([soon]), "session": ["amc"]})

    monkeypatch.setattr(planner, "get_earnings_dates", _earnings)
    ex_div = (now + pd.Timedelta(days=5)).tz_localize(None).normalize()
    monkeypatch.setattr(planner, "get_next_ex_dividend", lambda t: ex_div)

    out = planner.propose_levels("TEST", "long")

    ev = out["events"]
    assert ev["next_earnings"]["session"] == "amc"
    assert ev["next_earnings"]["days_until"] == 7
    assert ev["ex_dividend"]["days_until"] == 5
    assert len(ev["warnings"]) == 2
    assert any("earnings" in w for w in ev["warnings"])
    assert any("ex-dividend" in w for w in ev["warnings"])
    assert ev["notes"] == []


def test_propose_levels_distant_events_are_context_not_warnings(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    far = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=45)

    def _earnings(*a, **k) -> pd.DataFrame:
        return pd.DataFrame({"earnings_datetime": pd.DatetimeIndex([far]), "session": ["bmo"]})

    monkeypatch.setattr(planner, "get_earnings_dates", _earnings)
    monkeypatch.setattr(planner, "get_next_ex_dividend", lambda t: None)

    out = planner.propose_levels("TEST", "long")

    ev = out["events"]
    assert ev["next_earnings"]["days_until"] == 45
    assert ev["ex_dividend"] is None
    assert ev["warnings"] == []


def test_propose_levels_event_failures_are_notes_not_fatal(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)

    def _boom(*a, **k):
        raise ConnectionError("rate limited")

    monkeypatch.setattr(planner, "get_earnings_dates", _boom)
    monkeypatch.setattr(planner, "get_next_ex_dividend", _boom)

    out = planner.propose_levels("TEST", "long")

    assert out["scenarios"]  # the proposal still stands
    assert out["events"]["next_earnings"] is None
    assert out["events"]["ex_dividend"] is None
    assert len(out["events"]["notes"]) == 2


def test_propose_levels_sparkline_and_context(monkeypatch) -> None:
    bars = _bars()
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: bars)
    _quiet_events(monkeypatch)

    out = planner.propose_levels("TEST", "long")

    assert len(out["sparkline"]["closes"]) == 60
    assert out["sparkline"]["closes"][-1] == out["spot"]
    assert out["sparkline"]["end"] == out["asof"]
    assert out["sparkline"]["start"] == bars.index[-60].strftime("%Y-%m-%d")
    assert out["context"]["high_20d"] == pytest.approx(
        float(bars["high"].iloc[-20:].max()), abs=0.01
    )
    assert out["context"]["low_20d"] == pytest.approx(float(bars["low"].iloc[-20:].min()), abs=0.01)


def test_hypothesis_ticket_chain_failure_propagates_when_fallback_also_down(monkeypatch) -> None:
    def _boom(t):
        raise ConnectionError("yfinance 429")

    def _cboe_down(t):
        raise ConnectionError("cdn offline")

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", _boom)
    monkeypatch.setattr(planner, "fetch_cboe_chain", _cboe_down)

    with pytest.raises(ConnectionError, match="yfinance 429"):  # the ORIGINAL error
        planner.hypothesis_ticket(
            ticker="TEST",
            stance="long",
            levels=dict(LEVELS),
            account_size=100_000.0,
            risk_per_trade_pct=0.01,
        )


# ── after-hours fallback: degenerate yfinance chain -> CBOE delayed quotes ──


def _zeroed(chain: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's overnight artifact: the board is listed but bid/ask/OI are zeroed."""
    out = chain.copy()
    out[["bid", "ask", "open_interest"]] = 0.0
    return out


def _ticket(**overrides) -> dict:
    kwargs: dict = {
        "ticker": "TEST",
        "stance": "long",
        "levels": dict(LEVELS),
        "account_size": 100_000.0,
        "risk_per_trade_pct": 0.01,
    }
    kwargs.update(overrides)
    return planner.hypothesis_ticket(**kwargs)


def test_hypothesis_ticket_zeroed_yf_chain_falls_back_to_cboe(make_chain, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", lambda t: _zeroed(make_chain(mids=MIDS)))
    monkeypatch.setattr(planner, "fetch_cboe_chain", lambda t: make_chain(mids=MIDS))
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    ticket = _ticket()

    assert any("CBOE delayed" in w for w in ticket["warnings"])
    assert all(
        s["calculator_url"].startswith("https://optionstrat.com/build/custom/TEST/")
        for s in ticket["structures"]
    )


def test_hypothesis_ticket_healthy_chain_never_calls_cboe(make_chain, monkeypatch) -> None:
    def _must_not_call(t):
        raise AssertionError("CBOE fallback must not fire on a healthy chain")

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", lambda t: make_chain(mids=MIDS))
    monkeypatch.setattr(planner, "fetch_cboe_chain", _must_not_call)
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    ticket = _ticket()

    assert not any("CBOE" in w for w in ticket["warnings"])


def test_hypothesis_ticket_zeroed_chain_and_cboe_down_keeps_original_error(
    make_chain, monkeypatch
) -> None:
    def _cboe_down(t):
        raise ConnectionError("cdn offline")

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", lambda t: _zeroed(make_chain(mids=MIDS)))
    monkeypatch.setattr(planner, "fetch_cboe_chain", _cboe_down)
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    with pytest.raises(ValueError, match="liquidity gate"):
        _ticket()


def test_hypothesis_ticket_yf_fetch_error_rescued_by_cboe(make_chain, monkeypatch) -> None:
    def _boom(t):
        raise ConnectionError("yfinance 429")

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", _boom)
    monkeypatch.setattr(planner, "fetch_cboe_chain", lambda t: make_chain(mids=MIDS))
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    ticket = _ticket()

    assert any("CBOE delayed" in w for w in ticket["warnings"])


def test_hypothesis_ticket_empty_yf_chain_uses_fallback(make_chain, monkeypatch) -> None:
    from tradinglib.loaders.options.yf_chain import CHAIN_COLUMNS

    monkeypatch.setattr(planner, "load_daily", lambda *a, **k: _bars())
    monkeypatch.setattr(planner, "fetch_chain", lambda t: pd.DataFrame(columns=CHAIN_COLUMNS))
    monkeypatch.setattr(planner, "fetch_cboe_chain", lambda t: make_chain(mids=MIDS))
    monkeypatch.setattr(planner, "get_earnings_dates", _no_earnings)

    ticket = _ticket()

    assert any("CBOE delayed" in w for w in ticket["warnings"])
