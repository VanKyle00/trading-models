"""Conformance harness over the tournament strategy registry.

Every registered strategy — current and future, rule-based or ML — must pass
these contracts. Causality is the load-bearing check: a signal value at bar t
must be identical when bars after t are removed, which mechanically catches
centered windows, full-sample normalization, and fit-on-test leakage. New
strategies inherit every check via the registry parametrization; no test
writing required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.tournament.levels import direction
from tradinglib.tournament.strategies import STRATEGIES
from tradinglib.validation.search import expand_grid

TRAIN_BARS = 500
TOTAL_BARS = 700


def synthetic_bars(n: int = TOTAL_BARS, seed: int = 7) -> pd.DataFrame:
    """Seeded random-walk OHLCV with an ``earnings`` flag column.

    The earnings column is unused by today's strategies; it is part of the
    fixture contract so event-driven strategies (sub-project B) exercise the
    same harness unchanged.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    high = close * (1.0 + rng.uniform(0.0, 0.02, n))
    low = close * (1.0 - rng.uniform(0.0, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    idx = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
    earnings = pd.Series(False, index=idx)
    earnings.iloc[60::63] = True  # quarterly-ish cadence
    bars["earnings"] = earnings
    return bars


_BARS = synthetic_bars()

CASES = [
    pytest.param(key, params, stance, id=f"{key}[{i}]-{stance}")
    for key, sdef in STRATEGIES.items()
    for i, params in enumerate(expand_grid(sdef.param_grid))
    for stance in ("long", "short")
]


@pytest.mark.parametrize(("key", "params", "stance"), CASES)
def test_signal_is_causal(key: str, params: dict, stance: str) -> None:
    # Truncating future bars must not change past signal values.
    sdef = STRATEGIES[key]
    train, test = _BARS.iloc[:TRAIN_BARS], _BARS.iloc[TRAIN_BARS:]
    full = sdef.make_signal(train, test, params, stance)
    for k in (10, 60, 150):
        part = sdef.make_signal(train, test.iloc[:k], params, stance)
        pd.testing.assert_series_equal(part, full.iloc[:k], check_names=False)


@pytest.mark.parametrize(("key", "params", "stance"), CASES)
def test_signal_contract(key: str, params: dict, stance: str) -> None:
    # Test-indexed output; {0, +1} long, {0, -1} short.
    sdef = STRATEGIES[key]
    train, test = _BARS.iloc[:TRAIN_BARS], _BARS.iloc[TRAIN_BARS:]
    sig = sdef.make_signal(train, test, params, stance)
    assert sig.index.equals(test.index)
    allowed = {0.0, 1.0} if direction(stance) > 0 else {-1.0, 0.0}
    assert set(np.unique(sig)) <= allowed


@pytest.mark.parametrize(("key", "params", "stance"), CASES)
def test_in_sample_mode(key: str, params: dict, stance: str) -> None:
    # walk_forward scores parameters via make_signal(train, train, params).
    sdef = STRATEGIES[key]
    train = _BARS.iloc[:TRAIN_BARS]
    sig = sdef.make_signal(train, train, params, stance)
    assert sig.index.equals(train.index)


@pytest.mark.parametrize(("key", "params", "stance"), CASES)
def test_determinism(key: str, params: dict, stance: str) -> None:
    # Identical inputs -> identical output (ML strategies must seed).
    sdef = STRATEGIES[key]
    train, test = _BARS.iloc[:TRAIN_BARS], _BARS.iloc[TRAIN_BARS:]
    a = sdef.make_signal(train, test, params, stance)
    b = sdef.make_signal(train, test, params, stance)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize(("key", "params", "stance"), CASES)
def test_levels_contract(key: str, params: dict, stance: str) -> None:
    # Levels (or None = "no actionable entry tonight"); stop on the loss side,
    # target on the profit side; input bars never mutated.
    sdef = STRATEGIES[key]
    before = _BARS.copy(deep=True)
    lv = sdef.levels(_BARS, params, stance)
    pd.testing.assert_frame_equal(_BARS, before)
    if lv is None:
        return
    d = direction(stance)
    assert lv.entry_type in ("market", "stop", "limit")
    assert d * (lv.entry - lv.stop) > 0
    assert d * (lv.target - lv.entry) > 0
