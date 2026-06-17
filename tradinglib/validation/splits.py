"""Walk-forward window generators.

Produce ``(train_index, test_index)`` pairs over a time-ordered index. Anchored
windows grow the training span from a fixed start; rolling windows slide a
fixed-size training span. With the default ``step == test_size`` the test
segments are non-overlapping and tile the evaluable span exactly once.
"""

from __future__ import annotations

import pandas as pd


def anchored_windows(
    index: pd.Index,
    initial_train: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-train walk-forward windows.

    ``embargo`` purges the last ``embargo`` train bars of every window, opening
    a gap before the test window (CON-02 purged CV): a label that looks up to
    ``embargo`` bars ahead then cannot bleed across the train/test boundary.
    ``embargo=0`` (the default) leaves windows unchanged.
    """
    if initial_train < 1 or test_size < 1:
        raise ValueError("initial_train and test_size must be >= 1")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if embargo >= initial_train:
        raise ValueError(f"embargo {embargo} must be < initial_train {initial_train}")

    windows: list[tuple[pd.Index, pd.Index]] = []
    n = len(index)
    train_end = initial_train
    while train_end + test_size <= n:
        windows.append((index[: train_end - embargo], index[train_end : train_end + test_size]))
        train_end += step
    return windows


def rolling_windows(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
) -> list[tuple[pd.Index, pd.Index]]:
    """Fixed-size sliding-train walk-forward windows.

    ``embargo`` purges the last ``embargo`` train bars of every window, opening
    a gap before the test window (CON-02 purged CV). ``embargo=0`` (the default)
    leaves windows unchanged.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if embargo >= train_size:
        raise ValueError(f"embargo {embargo} must be < train_size {train_size}")

    windows: list[tuple[pd.Index, pd.Index]] = []
    n = len(index)
    start = 0
    while start + train_size + test_size <= n:
        s = start + train_size
        windows.append((index[start : s - embargo], index[s : s + test_size]))
        start += step
    return windows
