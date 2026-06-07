"""Tests for the walk-forward harness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.validation.walk_forward import WalkForwardResult, walk_forward


def _data(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)  # steady uptrend
    return pd.DataFrame({"close": close, "open": close}, index=idx)


def _make_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.Series:
    # "long" param True => fully invested; False => flat.
    value = 1.0 if params["long"] else 0.0
    return pd.Series(value, index=test.index)


def test_walk_forward_selects_best_param_and_deflates() -> None:
    data = _data()
    grid = {"long": [True, False]}
    res = walk_forward(
        data, _make_signal, param_grid=grid, mode="anchored",
        initial_train=40, test_size=20,
    )
    assert isinstance(res, WalkForwardResult)
    # On a pure uptrend, "long" wins every window.
    assert (res.windows["param_long"] == True).all()  # noqa: E712
    # OOS index is the union of test windows; n_trials == grid size.
    assert res.oos_result.config["n_trials"] == 2
    assert res.oos_result.equity_curve.iloc[-1] > 100_000.0
    assert "long" in res.param_stability


def test_walk_forward_rejects_misindexed_signal() -> None:
    data = _data()

    def bad_signal(train, test, params):
        return pd.Series(1.0, index=test.index[:-1])  # wrong length

    with pytest.raises(ValueError, match="indexed like test"):
        walk_forward(data, bad_signal, param_grid={"long": [True]}, mode="anchored",
                     initial_train=40, test_size=20)


def test_walk_forward_requires_window_sizing() -> None:
    data = _data()
    with pytest.raises(ValueError, match="initial_train"):
        walk_forward(data, _make_signal, param_grid={"long": [True]}, mode="anchored", test_size=20)
