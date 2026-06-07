"""Tests for walk-forward window generators."""
from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.validation.splits import anchored_windows, rolling_windows


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_anchored_grows_train_and_tiles_test() -> None:
    idx = _idx(100)
    wins = anchored_windows(idx, initial_train=10, test_size=5)
    assert len(wins) == (100 - 10) // 5
    assert len(wins[0][0]) == 10 and len(wins[1][0]) == 15  # train grows
    assert all(len(test) == 5 for _, test in wins)
    assert wins[0][0].intersection(wins[0][1]).empty
    assert wins[0][1].max() < wins[1][1].min()


def test_rolling_keeps_train_fixed() -> None:
    idx = _idx(100)
    wins = rolling_windows(idx, train_size=10, test_size=5)
    assert all(len(train) == 10 for train, _ in wins)
    assert wins[0][0].min() < wins[1][0].min()  # window slides
    assert wins[0][0].intersection(wins[0][1]).empty


def test_invalid_sizes_raise() -> None:
    idx = _idx(10)
    with pytest.raises(ValueError):
        anchored_windows(idx, initial_train=0, test_size=5)
    with pytest.raises(ValueError):
        rolling_windows(idx, train_size=5, test_size=0)
