"""Dispatch a model to its ``run_for_gui`` function.

Each ``models/<family>/<slug>/backtest.py`` exposes ``run_for_gui(start,
end, **params)``. This module loads that file dynamically (the slugs
contain hyphens and leading digits, so a regular ``import`` is not
possible) and forwards the call.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from tradinglib.data.paths import repo_root


def _load_backtest_module(model_dir: Path) -> ModuleType:
    backtest_file = model_dir / "backtest.py"
    if not backtest_file.exists():
        raise FileNotFoundError(f"no backtest.py at {model_dir}")
    # Make the model dir importable so backtest.py can import sibling modules
    # (e.g. the XGBoost model's backtest.py imports its train.py).
    sys.path.insert(0, str(model_dir))
    try:
        module_name = f"_model_backtest_{model_dir.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, backtest_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not build import spec for {backtest_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def run(model_meta: dict[str, Any], start: Any, end: Any, **params: Any) -> dict[str, Any]:
    """Invoke the model's ``run_for_gui`` and return its result dict."""
    model_dir = repo_root() / model_meta["_path"]
    module = _load_backtest_module(model_dir)
    if not hasattr(module, "run_for_gui"):
        raise AttributeError(
            f"{model_dir}/backtest.py has no run_for_gui() — the model needs "
            "to be refactored before the GUI can run it"
        )
    return module.run_for_gui(start, end, **params)
