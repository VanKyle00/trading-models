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
        data,
        _make_signal,
        param_grid=grid,
        mode="anchored",
        initial_train=40,
        test_size=20,
    )
    assert isinstance(res, WalkForwardResult)
    # On a pure uptrend, "long" wins every window.
    assert (res.windows["param_long"] == True).all()  # noqa: E712
    # OOS index is the union of test windows; n_trials == grid size.
    assert res.oos_result.config["n_trials"] == 2
    assert res.oos_result.equity_curve.iloc[-1] > 100_000.0
    assert "long" in res.param_stability


def test_walk_forward_rejects_misindexed_oos_signal() -> None:
    data = _data()

    def bad_signal(train, test, params):
        # Correct in-sample (train == test); wrong only on the real OOS slice,
        # so the out-of-sample guard is what fires.
        if test.index.equals(train.index):
            return pd.Series(1.0, index=test.index)
        return pd.Series(1.0, index=test.index[:-1])

    with pytest.raises(ValueError, match="indexed like test"):
        walk_forward(
            data,
            bad_signal,
            param_grid={"long": [True]},
            mode="anchored",
            initial_train=40,
            test_size=20,
        )


def test_walk_forward_rejects_misindexed_in_sample_signal() -> None:
    data = _data()

    def bad_in_sample(train, test, params):
        return pd.Series(1.0, index=test.index[:-1])  # wrong length even in-sample

    with pytest.raises(ValueError, match="indexed like the in-sample window"):
        walk_forward(
            data,
            bad_in_sample,
            param_grid={"long": [True]},
            mode="anchored",
            initial_train=40,
            test_size=20,
        )


def test_walk_forward_requires_window_sizing() -> None:
    data = _data()
    with pytest.raises(ValueError, match="initial_train"):
        walk_forward(data, _make_signal, param_grid={"long": [True]}, mode="anchored", test_size=20)


def test_walk_forward_rolling_mode() -> None:
    data = _data()
    res = walk_forward(
        data,
        _make_signal,
        param_grid={"long": [True, False]},
        mode="rolling",
        train_size=40,
        test_size=20,
    )
    assert isinstance(res, WalkForwardResult)
    assert res.oos_result.config["n_trials"] == 2
    assert (res.windows["param_long"] == True).all()  # noqa: E712


def test_walk_forward_embargo_withholds_boundary_bars_from_the_signal() -> None:
    # With embargo=H the signal builder must never be handed the H train bars
    # immediately before a test window — that is the purge that stops a
    # forward-looking training label from leaking across the boundary.
    data = _data()
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def recording_signal(train, test, params):
        if not test.index.equals(train.index):  # real OOS call, not in-sample scoring
            seen.append((train.index.max(), test.index.min()))
        return pd.Series(0.0, index=test.index)

    walk_forward(
        data,
        recording_signal,
        param_grid={"long": [True]},
        mode="anchored",
        initial_train=40,
        test_size=20,
        embargo=4,
    )
    assert seen  # at least one OOS window ran
    for train_max, test_min in seen:
        gap = data.index[(data.index > train_max) & (data.index < test_min)]
        assert len(gap) == 4  # exactly the embargo is withheld from the model


def test_walk_forward_embargo_zero_is_unchanged() -> None:
    data = _data()
    kw = dict(param_grid={"long": [True, False]}, mode="anchored", initial_train=40, test_size=20)
    base = walk_forward(data, _make_signal, **kw)
    emb0 = walk_forward(data, _make_signal, **kw, embargo=0)
    pd.testing.assert_series_equal(base.oos_result.returns, emb0.oos_result.returns)


def test_walk_forward_rejects_empty_search_space() -> None:
    data = _data()
    with pytest.raises(ValueError, match="empty search space"):
        walk_forward(
            data,
            _make_signal,
            param_grid={"long": []},
            mode="anchored",
            initial_train=40,
            test_size=20,
        )
