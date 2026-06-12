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


def test_certify_requires_min_dates_and_dsr() -> None:
    from tradinglib.scanner.pooled import certify

    # Seeded normal series: mean≈0.49, std≈0.16 → per-obs SR≈3 → DSR≈1.0 with n_trials=6.
    # The plan's alternating [0.8, 0.6] series is bimodal (excess_kurt≈-2.17), which makes
    # the Lo/BLdP sr_var go negative → compute_metrics short-circuits to DSR 0.0.
    # Using a seeded normal series preserves the intent (strongly positive, 25 dates).
    rng = np.random.default_rng(42)
    strong = pd.Series(
        0.5 + 0.2 * rng.standard_normal(25),
        index=pd.date_range("2025-01-01", periods=25, freq="W"),
    )
    thin = strong.iloc[:5]
    verdicts = certify(
        {("pead", "long"): strong, ("base_breakout", "long"): thin},
        n_trials=6,
        min_dates=20,
        fdr_alpha=0.10,
    )
    assert verdicts[("pead", "long")]["certified"] is True
    assert verdicts[("base_breakout", "long")]["certified"] is False
    assert "n_dates 5 < 20" in verdicts[("base_breakout", "long")]["reasons"]


def test_certify_dsr_block_and_fdr_informational() -> None:
    from tradinglib.scanner.pooled import certify

    rng = np.random.default_rng(42)
    strong = pd.Series(
        rng.normal(0.5, 0.2, 25), index=pd.date_range("2025-01-01", periods=25, freq="W")
    )
    weak = pd.Series(
        rng.normal(-0.05, 0.5, 25), index=pd.date_range("2025-01-01", periods=25, freq="W")
    )

    # DSR floor blocks even when the FDR pass is generous
    verdicts = certify({("pead", "long"): weak}, n_trials=1, min_dates=20, fdr_alpha=0.99)
    v = verdicts[("pead", "long")]
    assert v["certified"] is False
    assert any(r.startswith("pooled_dsr") for r in v["reasons"])

    # FDR failure on an otherwise-clean key is informational-only
    verdicts = certify({("pead", "long"): strong}, n_trials=6, min_dates=20, fdr_alpha=0.0)
    v = verdicts[("pead", "long")]
    assert v["certified"] is False
    assert v["reasons"] == ["failed pooled FDR"]


def test_build_certification_shape() -> None:
    from tradinglib.scanner.config import ScanConfig
    from tradinglib.scanner.pooled import build_certification

    bars = {
        "AAA": _trending_bars("AAA"),
        "BBB": _trending_bars("BBB", drift=-0.001),
        "BAIT": _bait_breakout_bars(),
    }
    sidecar = build_certification(bars, {}, asof=pd.Timestamp("2026-01-30"), config=ScanConfig())
    assert sidecar["built_asof"] == "2026-01-30"
    assert sidecar["lookback_days"] == 750
    assert sidecar["sim_errors"] == 0
    assert "pead:long" in sidecar["verdicts"]
    assert "base_breakdown:short" in sidecar["verdicts"]
    assert all(
        {"certified", "pooled_dsr", "n_dates", "total_r", "reasons"} <= set(v)
        for v in sidecar["verdicts"].values()
    )
    assert len(sidecar["verdicts"]) == 6  # whole menu: 3 long types + 3 short types
    assert sidecar["verdicts"]["base_breakout:long"]["n_dates"] > 0  # bait drives a real series
