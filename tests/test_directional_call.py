"""The directional-call demo runs frictionless vs realistic and costs money under frictions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/options/02-directional-call-spy"


def _load_module():
    spec = importlib.util.spec_from_file_location("_directional_call", MODEL_DIR / "backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spread_reduces_equity() -> None:
    mod = _load_module()
    idx = pd.date_range("2023-01-01", periods=180, freq="B")
    prices = pd.Series(100.0 * (1.0002 ** np.arange(180)), index=idx)  # gentle uptrend
    out = mod.run_compare(prices, tenor_days=60, otm_pct=0.0)
    assert set(out) >= {"naive_flat", "surface_no_spread", "surface_with_spread"}
    # Holding the surface fixed, adding the spread can only cost money (same fills/marks).
    assert (
        out["surface_with_spread"].equity_curve.iloc[-1]
        <= out["surface_no_spread"].equity_curve.iloc[-1]
    )
