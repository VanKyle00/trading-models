# tradinglib/service/__init__.py
"""Frontend-agnostic backtest service: typed request → validated run → result.

Exports are built up across the substrate plan: spec symbols land first (Task
3), request symbols in Task 4, run symbols in Task 6. Keep imports limited to
modules that already exist so the package imports cleanly at every stage.
"""

from __future__ import annotations

from tradinglib.service.request import BacktestRequest, RequestError, validate
from tradinglib.service.spec import ModelSpec, ParamSpec, list_specs, model_spec

__all__ = [
    "BacktestRequest",
    "ModelSpec",
    "ParamSpec",
    "RequestError",
    "list_specs",
    "model_spec",
    "validate",
]
