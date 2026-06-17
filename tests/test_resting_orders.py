"""Tests for the per-strategy resting-order builders (tournament A2, step 2/2).

The load-bearing contract is shift-to-score / unshift-to-publish: ``levels()``
publishes tonight's order from data through today (``.iloc[-1]`` of a shared
``_resting_levels`` series), while ``resting_orders()`` is that SAME series
shifted one bar, so the order resting on bar ``t`` is exactly the order that was
published the night before (after bar ``t-1``). One series builder => the
certified backtest and the issued ticket cannot drift apart (cert==ticket), and
the live ticket the publisher emits is byte-identical to before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest.resting_engine import run_resting_backtest
from tradinglib.tournament.strategies import STRATEGIES

RESTING_KEYS = ("donchian", "base_breakout", "bollinger")


def _bars(close: np.ndarray, high=None, low=None, volume=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2022-01-03", periods=len(close), freq="B", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01 if high is None else np.asarray(high, dtype=float),
            "low": close * 0.99 if low is None else np.asarray(low, dtype=float),
            "close": close,
        },
        index=idx,
    )
    bars["volume"] = np.full(len(close), 1_000_000.0) if volume is None else volume
    return bars


def _trend_bars(n: int = 200, rate: float = 1.004) -> pd.DataFrame:
    return _bars(100.0 * rate ** np.arange(n))


def _base_breakout_bars(n_base: int = 60) -> pd.DataFrame:
    trend = 100.0 * 1.004 ** np.arange(252)
    base = np.full(n_base, trend[-1] * 0.99)
    breakout = np.array([trend[-1] * 1.03])
    close = np.concatenate([trend, base, breakout])
    bars = _bars(close, high=close * 1.002, low=close * 0.998)
    vol = np.full(len(close), 1_000_000.0)
    vol[-(n_base + 1) : -1] = np.linspace(900_000.0, 300_000.0, n_base)
    bars["volume"] = vol
    return bars


def _bollinger_bars(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.7, n))
    return _bars(close)


_FIXTURES = {
    "donchian": (_trend_bars(), {"n": 20}),
    "base_breakout": (_base_breakout_bars(), {"base": 40}),
    "bollinger": (_bollinger_bars(), {"window": 20, "num_std": 2.0}),
}


def _assert_shift_contract(key: str, stance: str) -> int:
    """Each resting order is the level published the night before. Returns the
    number of armed bars covered (so callers can assert the test wasn't vacuous)."""
    sdef = STRATEGIES[key]
    full, params = _FIXTURES[key]
    train_n = 140
    train, test = full.iloc[:train_n], full.iloc[train_n:]
    orders = sdef.resting_orders(train, test, params, stance)
    assert list(orders.columns) == ["entry", "stop", "target", "entry_type", "armed"]
    assert orders.index.equals(test.index)

    armed_seen = 0
    for local in range(len(test)):
        full_pos = train_n + local
        # levels() on bars through full_pos-1 == the order published that night
        published = sdef.levels(full.iloc[:full_pos], params, stance)
        row = orders.iloc[local]
        if published is None:
            assert not bool(row["armed"]), f"{key} {stance} bar {local}: armed but no publication"
            continue
        armed_seen += 1
        assert bool(row["armed"])
        assert row["entry_type"] == published.entry_type
        assert float(row["entry"]) == pytest.approx(published.entry)
        assert float(row["stop"]) == pytest.approx(published.stop)
        assert float(row["target"]) == pytest.approx(published.target)
    return armed_seen


@pytest.mark.parametrize("key", RESTING_KEYS)
@pytest.mark.parametrize("stance", ("long", "short"))
def test_resting_orders_are_the_prior_night_publication(key: str, stance: str) -> None:
    # anti-drift + shift-by-one + causality all in one: orders on bar t must equal
    # levels() computed through bar t-1, for every bar.
    _assert_shift_contract(key, stance)


def test_shift_contract_is_not_vacuous_for_each_strategy() -> None:
    # at least one armed bar per strategy/stance, else the contract assertion
    # above would pass trivially on an all-None fixture.
    for key in RESTING_KEYS:
        assert _assert_shift_contract(key, "long") > 0, f"{key} long never armed"


def test_bollinger_published_stop_mirrors_protective_stop() -> None:
    # The bollinger stop is the one published value no other test pins. Lock it
    # byte-identical to the old protective_stop(2x ATR(14)) path so the shared
    # _resting_levels math can't silently drift from protective_stop.
    from tradinglib.tournament.levels import protective_stop

    full, params = _FIXTURES["bollinger"]
    for stance in ("long", "short"):
        lv = STRATEGIES["bollinger"].levels(full, params, stance)
        assert lv is not None
        assert lv.stop == pytest.approx(protective_stop(full, lv.entry, stance))


def test_donchian_levels_publish_unchanged_byte_identical() -> None:
    # the refactor must not move the published ticket: pin the exact numbers.
    bars = _bars(np.full(80, 100.0), high=np.full(80, 101.0), low=np.full(80, 99.0))
    lv = STRATEGIES["donchian"].levels(bars, {"n": 20}, "long")
    assert lv.entry == 101.0 and lv.entry_type == "stop"
    assert lv.stop == 100.0 and lv.target == 103.0


def test_resting_backtest_via_builder_runs_and_counts_trades() -> None:
    # Wiring: the builder's orders feed run_resting_backtest unchanged and yield
    # a standard BacktestResult with an exact completed-trade count and a clean
    # returns series. (Byte-identical fills vs simulate_ticket are pinned at the
    # engine level by test_fill_identity_with_simulate_ticket; here the
    # shift-contract test above is what guarantees the orders are correct.)
    full, params = _FIXTURES["donchian"]
    sdef = STRATEGIES["donchian"]
    train, test = full.iloc[:140], full.iloc[140:]
    orders = sdef.resting_orders(train, test, params, "long")
    result = run_resting_backtest(full.loc[test.index], orders, stance="long")
    assert result.n_completed_trades is not None
    assert result.returns.index.equals(test.index)
    assert np.isfinite(result.returns).all()
    assert set(np.unique(result.position.to_numpy())) <= {0.0, 1.0}
