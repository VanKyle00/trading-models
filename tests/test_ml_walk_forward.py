"""The XGBoost walk-forward adapter fits on train and returns a test-indexed signal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/ml/01-gbm-next-day-return-spy"


def _load_module():
    sys.path.insert(0, str(MODEL_DIR))
    spec = importlib.util.spec_from_file_location("ml_walk_forward", MODEL_DIR / "walk_forward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_signal_fits_and_is_test_indexed() -> None:
    mod = _load_module()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=400, freq="D")
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, 400)), index=idx)
    data = pd.DataFrame({"close": close, "open": close}, index=idx)
    train, test = data.iloc[:300], data.iloc[300:]
    sig = mod.make_signal(train, test, {"max_depth": 3, "n_estimators": 50})
    assert sig.index.equals(test.index)
    assert set(sig.unique()) <= {0.0, 1.0}
