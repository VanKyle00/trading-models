"""Tests for the CBOE delayed-quotes chain loader (HTTP mocked, never live)."""

from __future__ import annotations

import urllib.error

import pandas as pd
import pytest


def _today_et() -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))


def _occ(root: str, exp: pd.Timestamp, right: str, strike: float) -> str:
    return f"{root}{exp:%y%m%d}{right}{int(strike * 1000):08d}"


def _record(symbol: str, *, bid: float = 1.0, ask: float = 1.1) -> dict:
    return {
        "option": symbol,
        "bid": bid,
        "ask": ask,
        "iv": 0.30,
        "open_interest": 500.0,
        "volume": 25.0,
    }


def _payload(options: list[dict], *, current_price: float = 100.0) -> dict:
    return {"data": {"current_price": current_price, "options": options}}


def test_fetch_maps_payload_to_canonical_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import cboe_chain
    from tradinglib.loaders.options.yf_chain import CHAIN_COLUMNS

    exp = _today_et() + pd.Timedelta(days=10)
    payload = _payload(
        [
            _record(_occ("AAPL", exp, "C", 95.0)),
            _record(_occ("AAPL", exp, "P", 90.5)),
        ],
        current_price=101.5,
    )
    monkeypatch.setattr(cboe_chain, "_get_json", lambda url: payload)

    df = cboe_chain.fetch_cboe_chain("AAPL")

    assert list(df.columns) == CHAIN_COLUMNS
    assert len(df) == 2
    assert set(df["right"]) == {"call", "put"}
    assert (df["expiration"] == exp).all()
    assert sorted(df["strike"]) == [90.5, 95.0]
    assert (df["spot"] == 101.5).all()
    assert (df["ticker"] == "AAPL").all()
    assert (df["date"] == _today_et()).all()
    assert (df["bid"] == 1.0).all() and (df["ask"] == 1.1).all()
    assert (df["iv"] == 0.30).all()
    assert (df["open_interest"] == 500.0).all() and (df["volume"] == 25.0).all()


def test_fetch_keeps_only_live_expirations_inside_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradinglib.loaders.options import cboe_chain

    today = _today_et()
    payload = _payload(
        [
            _record(_occ("AAPL", today - pd.Timedelta(days=1), "C", 100.0)),  # expired
            _record(_occ("AAPL", today + pd.Timedelta(days=10), "C", 100.0)),  # in window
            _record(_occ("AAPL", today + pd.Timedelta(days=200), "C", 100.0)),  # LEAP
        ]
    )
    monkeypatch.setattr(cboe_chain, "_get_json", lambda url: payload)

    df = cboe_chain.fetch_cboe_chain("AAPL")  # default window: 120 days

    assert df["expiration"].nunique() == 1
    assert (df["expiration"] == today + pd.Timedelta(days=10)).all()


def test_dashed_ticker_requested_with_dot_but_reported_with_dash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradinglib.loaders.options import cboe_chain

    exp = _today_et() + pd.Timedelta(days=10)
    seen: list[str] = []

    def fake_get(url: str) -> dict:
        seen.append(url)
        # CBOE strips the share-class dot in the per-record OCC symbol
        return _payload([_record(_occ("BRKB", exp, "C", 270.0))])

    monkeypatch.setattr(cboe_chain, "_get_json", fake_get)

    df = cboe_chain.fetch_cboe_chain("brk-b")

    assert seen == ["https://cdn.cboe.com/api/global/delayed_quotes/options/BRK.B.json"]
    assert (df["ticker"] == "BRK-B").all()
    assert (df["strike"] == 270.0).all()


def test_malformed_option_symbols_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import cboe_chain

    exp = _today_et() + pd.Timedelta(days=10)
    payload = _payload(
        [
            _record("GARBAGE"),
            _record(_occ("AAPL", exp, "C", 100.0)),
        ]
    )
    monkeypatch.setattr(cboe_chain, "_get_json", lambda url: payload)

    df = cboe_chain.fetch_cboe_chain("AAPL")

    assert len(df) == 1
    assert (df["strike"] == 100.0).all()


def test_fetch_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import cboe_chain

    def boom(url: str) -> dict:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(cboe_chain, "_get_json", boom)

    with pytest.raises(urllib.error.URLError):
        cboe_chain.fetch_cboe_chain("AAPL")
