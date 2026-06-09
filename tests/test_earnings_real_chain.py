"""Tests for the Phase-2 quote-to-quote real-chain event mechanics."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "options" / "03-earnings-straddle-spy"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"es_{name}", _MODEL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load("real_chain")
sig = _load("signal")


def _chain(rows: list[tuple]) -> pd.DataFrame:
    """rows: (expiration, strike, right, bid, ask)"""
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2026-03-02"),
            "ticker": "TEST",
            "expiration": [pd.Timestamp(r[0]) for r in rows],
            "strike": [float(r[1]) for r in rows],
            "right": [r[2] for r in rows],
            "bid": [float(r[3]) for r in rows],
            "ask": [float(r[4]) for r in rows],
            "iv": 0.30,
        }
    )


EARNINGS = pd.Timestamp("2026-03-05 21:05:00")  # AMC on Mar 5 (tz-naive)


# ---------- pick_expiry ----------


def test_pick_expiry_nearest_strictly_after_earnings() -> None:
    chain = _chain(
        [
            ("2026-03-05", 100, "call", 1, 2),  # ON the earnings date -> excluded (expires
            ("2026-03-13", 100, "call", 1, 2),  # before the AMC move is realized)
            ("2026-03-20", 100, "call", 1, 2),
        ]
    )
    assert rc.pick_expiry(chain, EARNINGS) == pd.Timestamp("2026-03-13")


def test_pick_expiry_none_when_all_expire_before() -> None:
    chain = _chain([("2026-03-04", 100, "call", 1, 2)])
    assert rc.pick_expiry(chain, EARNINGS) is None


def test_pick_expiry_none_on_empty_chain() -> None:
    assert rc.pick_expiry(_chain([]), EARNINGS) is None


# ---------- pick_atm_strike ----------


def test_pick_atm_strike_requires_both_legs_quoted() -> None:
    chain = _chain(
        [
            ("2026-03-13", 100, "call", 1.0, 1.2),  # call only -> not eligible
            ("2026-03-13", 105, "call", 1.0, 1.2),
            ("2026-03-13", 105, "put", 1.0, 1.2),  # both legs -> eligible
        ]
    )
    assert rc.pick_atm_strike(chain, pd.Timestamp("2026-03-13"), spot=100.0) == 105.0


def test_pick_atm_strike_zero_bid_disqualifies() -> None:
    chain = _chain(
        [
            ("2026-03-13", 100, "call", 0.0, 1.2),  # zero entry bid -> unquoted
            ("2026-03-13", 100, "put", 1.0, 1.2),
        ]
    )
    assert rc.pick_atm_strike(chain, pd.Timestamp("2026-03-13"), spot=100.0) is None


def test_pick_atm_strike_nearest_to_spot() -> None:
    rows = []
    for strike in (95, 100, 105):
        rows.append(("2026-03-13", strike, "call", 1.0, 1.2))
        rows.append(("2026-03-13", strike, "put", 1.0, 1.2))
    assert rc.pick_atm_strike(_chain(rows), pd.Timestamp("2026-03-13"), spot=101.0) == 100.0


# ---------- run_event ----------


def _bars() -> pd.Series:
    idx = pd.bdate_range("2026-02-23", "2026-03-10")
    return pd.Series(100.0, index=idx, name="close")


def _entry_chain(call_bid=4.8, call_ask=5.0, put_bid=3.8, put_ask=4.0) -> pd.DataFrame:
    return _chain(
        [
            ("2026-03-13", 100, "call", call_bid, call_ask),
            ("2026-03-13", 100, "put", put_bid, put_ask),
        ]
    )


def _exit_chain(call_bid=3.0, put_bid=2.0) -> pd.DataFrame:
    return _chain(
        [
            ("2026-03-13", 100, "call", call_bid, call_bid + 0.2),
            ("2026-03-13", 100, "put", put_bid, put_bid + 0.2),
        ]
    )


def _loader(entry: pd.DataFrame, exit_: pd.DataFrame):
    """Earnings bar for the Mar-5 AMC event is Mar 6 (first bar >= 21:05), so
    entry (3 bars before) is 2026-03-03 and exit (1 bar after) is 2026-03-09."""

    def load_chain(ticker: str, when, **kwargs) -> pd.DataFrame:
        return entry if pd.Timestamp(when) <= pd.Timestamp("2026-03-03") else exit_

    return load_chain


def _run(entry, exit_, **kwargs):
    defaults = dict(
        ticker="TEST",
        close=_bars(),
        earnings_datetime=EARNINGS,
        prior_moves=[0.10] * 8,  # expected_move = 0.10
        load_chain=_loader(entry, exit_),
    )
    defaults.update(kwargs)
    return rc.run_event(**defaults)


def test_run_event_quote_to_quote_pnl_arithmetic() -> None:
    rec = _run(_entry_chain(), _exit_chain())

    assert "skip_reason" not in rec
    assert rec["entry_cost"] == pytest.approx(9.0)  # call_ask 5.0 + put_ask 4.0
    assert rec["exit_value"] == pytest.approx(5.0)  # call_bid 3.0 + put_bid 2.0
    # fees = (9 + 5) * 100 * 1bp = 0.14 ; pnl = -400 - 0.14
    assert rec["pnl"] == pytest.approx(-400.14)
    # implied move from MIDs: ((4.9) + (3.9)) / 100 = 0.088
    assert rec["implied_move"] == pytest.approx(0.088)
    # gate: expected 0.10 > 0.088 * 1.2 = 0.1056 ? NO -> gate must not fire
    assert rec["gate_fired"] is False


def test_run_event_gate_fires_on_real_edge() -> None:
    rec = _run(_entry_chain(), _exit_chain(), prior_moves=[0.15] * 8)
    # expected 0.15 > 0.088 * 1.2 = 0.1056 -> fires
    assert rec["gate_fired"] is True


def test_run_event_nan_expected_trades_unfiltered_but_gate_off() -> None:
    rec = _run(_entry_chain(), _exit_chain(), prior_moves=[])
    assert "skip_reason" not in rec  # unfiltered branch keeps forecast-less events
    assert math.isnan(rec["expected_move"])
    assert rec["gate_fired"] is False


def test_run_event_zero_exit_bid_is_total_loss_not_skip() -> None:
    rec = _run(_entry_chain(), _exit_chain(call_bid=0.0, put_bid=0.0))
    assert "skip_reason" not in rec
    assert rec["exit_value"] == 0.0
    # fees = (9 + 0) * 100 * 1bp = 0.09 ; pnl = -900 - 0.09
    assert rec["pnl"] == pytest.approx(-900.09)


def test_run_event_skip_no_entry_chain() -> None:
    rec = _run(_chain([]), _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_NO_ENTRY_CHAIN


def test_run_event_skip_no_post_earnings_expiry() -> None:
    entry = _chain([("2026-03-04", 100, "call", 1, 2), ("2026-03-04", 100, "put", 1, 2)])
    rec = _run(entry, _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_NO_EXPIRY


def test_run_event_skip_no_quoted_atm() -> None:
    entry = _chain([("2026-03-13", 100, "call", 1, 2)])  # missing put leg
    rec = _run(entry, _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_NO_QUOTED_ATM


def test_run_event_skip_spread_over_cap() -> None:
    # bid/ask: call 1.0/3.0, put 1.0/3.0 -> mid premium 4.0, spread 4.0 -> frac 1.0 > 0.20
    entry = _chain([("2026-03-13", 100, "call", 1.0, 3.0), ("2026-03-13", 100, "put", 1.0, 3.0)])
    rec = _run(entry, _exit_chain())
    assert rec["skip_reason"] == rc.SKIP_SPREAD


def test_run_event_skip_no_exit_chain() -> None:
    rec = _run(_entry_chain(), _chain([]))
    assert rec["skip_reason"] == rc.SKIP_NO_EXIT_CHAIN


def test_run_event_window_out_of_range_raises() -> None:
    close = _bars().iloc[-4:]  # not enough bars before earnings for entry_lead=3
    with pytest.raises(ValueError, match="window"):
        _run(_entry_chain(), _exit_chain(), close=close)


# ---------- past_abs_moves consistency with signal.expected_move ----------


def test_past_abs_moves_matches_signal_expected_move() -> None:
    idx = pd.bdate_range("2025-01-01", periods=120)
    close = pd.Series(
        [100 + (i % 7) * 1.5 for i in range(120)], index=idx, name="close", dtype=float
    )
    earnings = pd.Series(pd.to_datetime(["2025-02-03 21:05", "2025-04-01 21:05"]))

    moves = rc.past_abs_moves(close, earnings)
    via_signal = sig.expected_move(close, earnings, lookback=8)

    assert len(moves) == 2
    assert pd.Series(moves[-8:]).mean() == pytest.approx(via_signal)


# ---------- gate_pnls sweep helper ----------


def test_gate_pnls_recomputes_gate_from_stored_rows() -> None:
    events = [
        {"implied_move": 0.05, "prior_moves": [0.10] * 8, "pnl": 100.0},  # 0.10 > 0.06 -> fires
        {"implied_move": 0.20, "prior_moves": [0.10] * 8, "pnl": -50.0},  # 0.10 < 0.24 -> no
    ]
    assert rc.gate_pnls(events, k=1.2, lookback=8) == [100.0]
    # tighter k excludes everything
    assert rc.gate_pnls(events, k=2.5, lookback=8) == []
