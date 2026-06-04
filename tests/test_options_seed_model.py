"""Smoke test for the delta-hedged-long-option seed model's GUI surface."""

from __future__ import annotations

import importlib.util

from tradinglib.backtest import BacktestResult
from tradinglib.data.paths import repo_root


def _load_model_module():
    model_dir = repo_root() / "models/options/01-delta-hedged-long-option-spy"
    spec = importlib.util.spec_from_file_location("_seed_delta_hedge", model_dir / "backtest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_for_gui_returns_expected_keys() -> None:
    module = _load_model_module()
    out = module.run_for_gui("2023-01-01", "2023-06-30", symbol="SPY", n_paths=200)
    assert isinstance(out["result"], BacktestResult)
    assert "close" in out["data"].columns
    assert out["simulation"].pnl_distribution.shape[0] <= 200
    assert "payoff" in out
    assert {"spots", "values", "expiry_values", "strike"} <= set(out["payoff"])
    assert out["params"]["symbol"] == "SPY"
    assert len(out["result"].equity_curve) == len(out["data"])
    assert out["symbol"] == "SPY"
