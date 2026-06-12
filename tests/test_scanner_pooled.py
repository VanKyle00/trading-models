"""Tests for pooled cross-sectional setup certification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.scanner.pooled import pooled_r_series, sweep_firings


def _trending_bars(symbol: str, n: int = 700, drift: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(sum(map(ord, symbol)) % 2**32)
    rets = drift + 0.01 * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": 1e6,
            "symbol": symbol,
        },
        index=idx,
    )


def test_sweep_firings_no_lookahead() -> None:
    bars = {"AAA": _trending_bars("AAA"), "BBB": _trending_bars("BBB")}
    firings = sweep_firings(
        bars,
        setup_types=("base_breakout",),
        stance="long",
        asof=pd.Timestamp("2026-01-30"),
        lookback_days=400,
        step_sessions=5,
        earnings_by_ticker={},
    )
    for f in firings:
        assert f["date"] <= "2026-01-30"
        assert f["setup_type"] == "base_breakout"
        assert {"ticker", "levels", "stance"} <= set(f)


def _bait_breakout_bars(n_ramp: int = 310, n_base: int = 50) -> pd.DataFrame:
    close = np.concatenate([np.linspace(50.0, 100.0, n_ramp), np.full(n_base, 100.0)])
    volume = np.concatenate([np.full(n_ramp, 1_000_000.0), np.full(n_base, 200_000.0)])
    idx = pd.date_range("2024-06-03", periods=n_ramp + n_base, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": volume,
            "symbol": "BAIT",
        },
        index=idx,
    )


def test_sweep_firings_bait_fires_and_round_trips() -> None:
    from tradinglib.strategist.evaluate import simulate_ticket

    bars = {"BAIT": _bait_breakout_bars(), "NOISE": _trending_bars("NOISE")}
    asof = bars["BAIT"].index[-1]
    firings = sweep_firings(
        bars,
        setup_types=("base_breakout",),
        stance="long",
        asof=asof.tz_localize("UTC"),
        lookback_days=120,
        step_sessions=5,
        earnings_by_ticker={},
    )
    bait = [f for f in firings if f["ticker"] == "BAIT"]
    assert bait, "bait fixture must fire the detector — vacuous sweep test"
    for f in bait:
        assert f["date"] <= asof.strftime("%Y-%m-%d")
        assert f["entry_window"] == 5  # base_breakout uses the default window
        assert f["levels"] is None or {"entry", "entry_type", "stop", "target"} <= set(f["levels"])
    fired = next(f for f in bait if f["levels"])
    result = simulate_ticket(
        fired, bars["BAIT"], asof=fired["date"], entry_window=fired["entry_window"]
    )
    assert "status" in result  # firing rows are valid simulate_ticket tickets


def test_pooled_r_series_aggregates_same_date() -> None:
    scored = [
        {"date": "2025-01-06", "r": 2.0, "status": "target"},
        {"date": "2025-01-06", "r": -1.0, "status": "stopped"},
        {"date": "2025-02-03", "r": -1.0, "status": "stopped"},
        {"date": "2025-03-03", "r": None, "status": "expired"},
    ]
    series = pooled_r_series(scored)
    assert list(series.index.strftime("%Y-%m-%d")) == ["2025-01-06", "2025-02-03"]
    assert series.iloc[0] == 0.5  # mean of the two same-date trades
