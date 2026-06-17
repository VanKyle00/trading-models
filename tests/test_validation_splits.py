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


def _gap(idx: pd.DatetimeIndex, train: pd.Index, test: pd.Index) -> pd.Index:
    return idx[(idx > train.max()) & (idx < test.min())]


def test_anchored_embargo_purges_a_gap_between_train_and_test() -> None:
    # A purge embargo removes the last `embargo` bars of train, opening a gap to
    # the test window so a label that looks `embargo` bars ahead cannot bleed
    # across the boundary (CON-02 purged CV). Test windows are untouched.
    idx = _idx(100)
    base = anchored_windows(idx, initial_train=20, test_size=5)
    emb = anchored_windows(idx, initial_train=20, test_size=5, embargo=3)
    assert len(emb) == len(base)
    for (tr, te), (btr, bte) in zip(emb, base, strict=True):
        assert te.equals(bte)  # test windows unchanged by embargo
        assert len(tr) == len(btr) - 3  # train shorter by exactly the embargo
        assert len(_gap(idx, tr, te)) == 3  # a 3-bar purge gap opens up
        assert tr.intersection(te).empty


def test_rolling_embargo_purges_a_gap_between_train_and_test() -> None:
    idx = _idx(100)
    base = rolling_windows(idx, train_size=20, test_size=5)
    emb = rolling_windows(idx, train_size=20, test_size=5, embargo=3)
    assert len(emb) == len(base)
    for (tr, te), (btr, bte) in zip(emb, base, strict=True):
        assert te.equals(bte)
        assert len(tr) == len(btr) - 3
        assert len(_gap(idx, tr, te)) == 3


def test_embargo_zero_matches_no_embargo() -> None:
    # The default (embargo=0) must produce byte-identical windows so existing
    # tournament/promotion results are unchanged.
    idx = _idx(100)
    for gen, kw in (
        (anchored_windows, {"initial_train": 10, "test_size": 5}),
        (rolling_windows, {"train_size": 10, "test_size": 5}),
    ):
        a = gen(idx, **kw)
        b = gen(idx, **kw, embargo=0)
        assert len(a) == len(b)
        for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
            assert tr_a.equals(tr_b) and te_a.equals(te_b)


def test_negative_embargo_raises() -> None:
    idx = _idx(50)
    with pytest.raises(ValueError):
        anchored_windows(idx, initial_train=20, test_size=5, embargo=-1)
    with pytest.raises(ValueError):
        rolling_windows(idx, train_size=20, test_size=5, embargo=-1)


def test_embargo_not_smaller_than_train_raises() -> None:
    idx = _idx(50)
    with pytest.raises(ValueError):
        anchored_windows(idx, initial_train=10, test_size=5, embargo=10)
    with pytest.raises(ValueError):
        rolling_windows(idx, train_size=10, test_size=5, embargo=10)
