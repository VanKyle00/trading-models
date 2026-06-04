# tradinglib/service/__init__.py
"""Frontend-agnostic backtest service: typed request → validated run → result."""

from __future__ import annotations

from tradinglib.service.request import BacktestRequest, RequestError, validate
from tradinglib.service.run import BacktestRun, run, run_to_dict
from tradinglib.service.spec import ModelSpec, ParamSpec, list_specs, model_spec

__all__ = [
    "BacktestRequest",
    "BacktestRun",
    "ModelSpec",
    "ParamSpec",
    "RequestError",
    "list_specs",
    "model_spec",
    "run",
    "run_to_dict",
    "validate",
]
