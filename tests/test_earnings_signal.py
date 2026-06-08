"""Tests for the earnings-straddle selection signal."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SIGNAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "options"
    / "03-earnings-straddle-spy"
    / "signal.py"
)
_spec = importlib.util.spec_from_file_location("earnings_signal", _SIGNAL_PATH)
assert _spec is not None and _spec.loader is not None
signal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(signal)


def test_signal_module_imports_math() -> None:
    # Task 4 Step 3 requires `math` in the top-of-module import block so Task 5's
    # passes_filter/tradeable_event (which call math.isnan) never insert a mid-file
    # import (which would violate ruff `I` sorting).
    import math as _math

    assert getattr(signal, "math", None) is _math


def test_implied_move_is_straddle_over_spot() -> None:
    assert signal.implied_move(straddle_premium=6.0, spot=100.0) == pytest.approx(0.06)


def test_implied_move_rejects_nonpositive_spot() -> None:
    with pytest.raises(ValueError):
        signal.implied_move(straddle_premium=6.0, spot=0.0)


def test_expected_move_is_mean_of_past_earnings_day_abs_returns() -> None:
    idx = pd.date_range("2023-01-02", periods=400, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    e_dates = [idx[50], idx[150], idx[250]]
    moves = [0.05, -0.03, 0.07]
    for ed, mv in zip(e_dates, moves, strict=True):
        pos = idx.get_loc(ed)
        close.iloc[pos + 1] = close.iloc[pos] * (1 + mv)

    em = signal.expected_move(close=close, earnings_datetimes=pd.Series(e_dates), lookback=3)
    # mean of |0.05|, |0.03|, |0.07| = 0.05
    assert em == pytest.approx(0.05, abs=1e-6)


def test_expected_move_tz_robust_naive_index_aware_earnings() -> None:
    idx = pd.date_range("2023-01-02", periods=400, freq="B")  # tz-NAIVE bars
    close = pd.Series(100.0, index=idx)
    pos = 50
    close.iloc[pos + 1] = close.iloc[pos] * 1.06
    earnings = pd.Series([pd.Timestamp(idx[pos], tz="UTC")])  # tz-AWARE earnings
    em = signal.expected_move(close=close, earnings_datetimes=earnings, lookback=8)
    assert em == pytest.approx(0.06, abs=1e-6)


def test_expected_move_nan_when_no_history() -> None:
    idx = pd.date_range("2023-01-02", periods=10, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    em = signal.expected_move(
        close=close, earnings_datetimes=pd.Series([], dtype="datetime64[ns, UTC]"), lookback=3
    )
    assert np.isnan(em)


def test_passes_filter_true_only_when_expected_beats_implied_times_k() -> None:
    assert signal.passes_filter(expected=0.10, implied=0.06, k=1.5) is True
    assert signal.passes_filter(expected=0.10, implied=0.06, k=1.7) is False


def test_passes_filter_rejects_k_le_one() -> None:
    with pytest.raises(ValueError):
        signal.passes_filter(expected=0.10, implied=0.06, k=1.0)


def test_passes_filter_false_on_nan_inputs() -> None:
    assert signal.passes_filter(expected=float("nan"), implied=0.06, k=1.2) is False
    assert signal.passes_filter(expected=0.10, implied=float("nan"), k=1.2) is False


def test_tradeable_event_rejects_bad_chains() -> None:
    base = dict(
        implied=0.06, expected=0.10, spread_frac=0.05, max_spread_frac=0.20, has_expiry=True
    )
    assert signal.tradeable_event(**base) is True
    assert signal.tradeable_event(**{**base, "implied": float("nan")}) is False
    assert signal.tradeable_event(**{**base, "spread_frac": 0.25}) is False
    assert signal.tradeable_event(**{**base, "has_expiry": False}) is False
