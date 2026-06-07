"""The SMA walk-forward adapter returns a test-indexed 0/1 signal."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/classical/01-sma-crossover-spy"


def _load_module():
    sys.path.insert(0, str(MODEL_DIR))
    spec = importlib.util.spec_from_file_location("sma_walk_forward", MODEL_DIR / "walk_forward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_signal_is_test_indexed_binary() -> None:
    mod = _load_module()
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    close = pd.Series(100.0 * (1.01 ** np.arange(300)), index=idx)
    data = pd.DataFrame({"close": close, "open": close}, index=idx)
    train, test = data.iloc[:250], data.iloc[250:]
    sig = mod.make_signal(train, test, {"fast": 10, "slow": 50})
    assert sig.index.equals(test.index)
    assert set(sig.unique()) <= {0.0, 1.0}
