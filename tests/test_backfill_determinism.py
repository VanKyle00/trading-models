"""Airtight A/B guardrail for the replay (audit item 4).

The model-testing harness is only trustworthy if a replayed night is a PURE
function of its bars: same inputs -> byte-identical issued levels. And the two
A/B arms (e.g. regime gate off vs on) must differ ONLY by which rows are
gated/promoted — never by the bars or the level math. These two tests lock in
both properties so a future change that introduces hidden non-determinism, or
that lets a gate perturb levels, fails CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglib.scanner.config import ScanConfig

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_scan.py"
_spec = importlib.util.spec_from_file_location("backfill_scan", _SCRIPT)
backfill_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill_scan)


def _bars(seed: int, n: int = 540) -> pd.DataFrame:
    """Seeded synthetic daily OHLCV on a tz-naive UTC index (the shape run_night
    receives after the harness's _naive_utc preprocessing)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")  # tz-naive
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def _inputs():
    bars_by_ticker = {"AAA": _bars(1), "SPY": _bars(2)}
    earnings_by_ticker = {"AAA": pd.DatetimeIndex([])}
    family = {"long": ["AAA"], "short": []}
    asof = bars_by_ticker["AAA"].index[-1]
    return asof, family, bars_by_ticker, earnings_by_ticker


def test_run_night_is_deterministic() -> None:
    """Two identical replays of one night produce byte-identical output — proves
    the funnel/issued/levels carry no hidden non-determinism (RNG, dict order,
    mutable state) that would confound an A/B delta."""
    asof, family, bars, earnings = _inputs()
    config = ScanConfig()
    funnel_a, issued_a, errors_a = backfill_scan.run_night(asof, family, bars, earnings, config)
    funnel_b, issued_b, errors_b = backfill_scan.run_night(asof, family, bars, earnings, config)
    assert funnel_a == funnel_b
    assert issued_a == issued_b
    assert errors_a == errors_b


def test_regime_arm_differs_only_by_gated_rows() -> None:
    """The regime-gate arm is a pure filter: every row it keeps is identical
    (same levels) to the ungated arm, and its issued set is a subset — the gate
    changes membership, never the bars or the level math."""
    asof, family, bars, earnings = _inputs()
    _, ungated, _ = backfill_scan.run_night(asof, family, bars, earnings, ScanConfig())
    _, gated, _ = backfill_scan.run_night(
        asof, family, bars, earnings, ScanConfig(regime_gate=True)
    )

    def _key(row: dict) -> tuple:
        return (row["ticker"], row["stance"], row["strategy"], row["tier"])

    ungated_levels = {_key(r): r["levels"] for r in ungated}
    for row in gated:
        assert _key(row) in ungated_levels  # gated arm never invents a row
        assert row["levels"] == ungated_levels[_key(row)]  # nor perturbs its levels
    assert len(gated) <= len(ungated)
