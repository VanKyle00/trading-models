"""Tests for the tournament strategy registry — signal and levels math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.tournament.strategies import STRATEGIES


def _bars(
    close: np.ndarray, high: np.ndarray | None = None, low: np.ndarray | None = None
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2022-01-03", periods=len(close), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01 if high is None else np.asarray(high, dtype=float),
            "low": close * 0.99 if low is None else np.asarray(low, dtype=float),
            "close": close,
        },
        index=idx,
    )


def _trend_bars(n: int = 420, rate: float = 1.005) -> pd.DataFrame:
    return _bars(100.0 * rate ** np.arange(n))


def _flat_bars(n: int = 60, close: float = 100.0, spread: float = 1.0) -> pd.DataFrame:
    c = np.full(n, close)
    return _bars(c, high=c + spread, low=c - spread)


def test_sma_cross_long_rides_an_uptrend() -> None:
    bars = _trend_bars()
    train, test = bars.iloc[:300], bars.iloc[300:]
    sig = STRATEGIES["sma_cross"].make_signal(train, test, {"fast": 10, "slow": 50}, "long")
    assert sig.index.equals(test.index)
    assert set(np.unique(sig)) <= {0.0, 1.0}
    assert sig.iloc[-1] == 1.0  # fast SMA above slow on a steady rise


def test_sma_cross_short_is_the_mirror_rule_not_a_sign_flip() -> None:
    bars = _trend_bars(rate=0.995)  # downtrend
    train, test = bars.iloc[:300], bars.iloc[300:]
    short = STRATEGIES["sma_cross"].make_signal(train, test, {"fast": 10, "slow": 50}, "short")
    assert set(np.unique(short)) <= {-1.0, 0.0}
    assert short.iloc[-1] == -1.0
    long = STRATEGIES["sma_cross"].make_signal(train, test, {"fast": 10, "slow": 50}, "long")
    assert (long == 0.0).all()  # the long rule never fires in a downtrend


def test_make_signal_handles_in_sample_train_equals_test() -> None:
    # walk_forward scores in-sample via make_signal(train, train, params)
    bars = _trend_bars()
    train = bars.iloc[:300]
    sig = STRATEGIES["sma_cross"].make_signal(train, train, {"fast": 10, "slow": 50}, "long")
    assert sig.index.equals(train.index)  # concat deduped, no doubled index


def test_sma_cross_levels_atr_stop_and_2r_target() -> None:
    bars = _flat_bars()  # ATR(14) == 2.0
    lv = STRATEGIES["sma_cross"].levels(bars, {"fast": 10, "slow": 50}, "long")
    assert lv.entry == 100.0 and lv.entry_type == "market"
    assert lv.stop == 96.0  # entry - 2*ATR
    assert lv.target == 108.0  # entry + 2R
    short = STRATEGIES["sma_cross"].levels(bars, {"fast": 10, "slow": 50}, "short")
    assert short.stop == 104.0 and short.target == 92.0
