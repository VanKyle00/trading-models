"""Tests for sensitivity + regime diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.validation.sensitivity import (
    metrics_by_regime,
    parameter_sensitivity,
    vol_regime,
)


def _data(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx)
    return pd.DataFrame({"close": close, "open": close}, index=idx)


def _make_signal(train, test, params):
    return pd.Series(1.0 if params["long"] else 0.0, index=test.index)


def test_parameter_sensitivity_one_row_per_config() -> None:
    data = _data()
    grid = {"long": [True, False]}
    frame = parameter_sensitivity(
        data, _make_signal, grid,
        train_index=data.index[:60], test_index=data.index[60:],
    )
    assert len(frame) == 2
    assert {"long", "sharpe", "annualized_return", "max_drawdown"} <= set(frame.columns)


def test_vol_regime_labels_all_bars() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    rets = np.concatenate([rng.normal(0.0, 0.005, 60), rng.normal(0.0, 0.02, 60)])  # vol shift
    close = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx)
    labels = vol_regime(close, window=10, n_bins=3)
    assert labels.dropna().nunique() == 3
    assert len(labels) == len(close)


def test_metrics_by_regime_partitions_by_year() -> None:
    idx = pd.date_range("2020-06-01", periods=400, freq="D")  # spans 2020 + 2021
    returns = pd.Series(0.001, index=idx)
    equity = (1.0 + returns).cumprod() * 100_000.0
    frame = metrics_by_regime(returns, equity, by="year")
    assert set(frame.index) == {2020, 2021}
    assert frame["n_bars"].sum() == len(returns)
