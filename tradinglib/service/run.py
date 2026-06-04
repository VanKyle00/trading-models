# tradinglib/service/run.py
"""The one validated execution path + JSON serialization.

``run`` resolves the model spec, validates the request, dynamically loads the
model's ``backtest.py``, and calls its ``run_for_gui``. ``run_to_dict`` turns
the result into a JSON-safe, downsampled view consumed by both the web charts
and the LLM tool return. Top-level keys are append-only (additive contract).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from tradinglib.backtest.engine import BacktestResult
from tradinglib.data.paths import repo_root
from tradinglib.eval.trades import trades_from_position
from tradinglib.service.request import BacktestRequest, validate
from tradinglib.service.spec import ModelSpec, model_spec

_MAX_POINTS = 1000


@dataclass
class BacktestRun:
    model_id: str
    symbol: str
    params: dict[str, Any]
    result: BacktestResult
    data: pd.DataFrame
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. options simulation/payoff


def _load_backtest_module(model_dir: Path) -> ModuleType:
    backtest_file = model_dir / "backtest.py"
    if not backtest_file.exists():
        raise FileNotFoundError(f"no backtest.py at {model_dir}")
    # Make the model dir importable so backtest.py can import sibling modules
    # (e.g. the ML model's `from train import ...`). A model's own module-level
    # sys.path edits would desync a naive pop(0), so snapshot the whole list and
    # restore it — each run() leaves sys.path exactly as it found it.
    #
    # Caveat: sibling modules import under their bare name (e.g. `train`), so two
    # models that each ship a `train.py` would collide in sys.modules. Acceptable
    # while only the ML model has one; revisit if a second such model is added.
    saved_path = sys.path[:]
    sys.path.insert(0, str(model_dir))
    try:
        module_name = f"_model_backtest_{model_dir.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, backtest_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not build import spec for {backtest_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
    return module


def _kwargs(request: BacktestRequest, spec: ModelSpec) -> dict[str, Any]:
    kw: dict[str, Any] = dict(request.params)
    # validate() checks symbol legality but does NOT normalize; run() owns
    # normalization (strip/upper) and default-filling before dispatch.
    symbol = request.symbol if request.symbol is not None else spec.default_ticker
    kw["symbol"] = symbol.strip().upper()
    for name in ("fee_bps", "slippage_bps", "initial_capital", "size_mult"):
        val = getattr(request, name)
        if val is not None:
            kw[name] = val
    return kw


def run(request: BacktestRequest) -> BacktestRun:
    spec = model_spec(request.model_id)
    validate(request, spec)

    module = _load_backtest_module(repo_root() / spec.id)
    if not hasattr(module, "run_for_gui"):
        raise AttributeError(f"{spec.id}/backtest.py has no run_for_gui()")

    out = module.run_for_gui(str(request.start), str(request.end), **_kwargs(request, spec))
    known = {"data", "result", "symbol", "params"}
    return BacktestRun(
        model_id=spec.id,
        symbol=out.get("symbol", ""),
        params=out.get("params", {}),
        result=out["result"],
        data=out["data"],
        extra={k: v for k, v in out.items() if k not in known},
    )


def _jsonify(obj: Any) -> Any:
    """Recursively coerce numpy scalars / timestamps to JSON-native types.

    Keeps ``run_to_dict``'s JSON-safe guarantee robust even if a model returns
    numpy scalars or date objects in its params/metrics/config.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar → python scalar
        return obj.item()
    if hasattr(obj, "isoformat"):  # date / datetime / Timestamp
        return obj.isoformat()
    return str(obj)


def _series(s: pd.Series) -> dict[str, list]:
    """Downsample to <= _MAX_POINTS points, always preserving the final bar.

    The terminal value (final equity) is the number users read first, so it is
    appended if the even-spaced stride would otherwise skip it.
    """
    if len(s) > _MAX_POINTS:
        step = len(s) // _MAX_POINTS + 1
        sampled = s.iloc[::step]
        if sampled.index[-1] != s.index[-1]:
            sampled = pd.concat([sampled, s.iloc[[-1]]])
        s = sampled
    return {
        "index": [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in s.index],
        "values": [None if pd.isna(v) else float(v) for v in s.values],
    }


def _iso(t: Any) -> str:
    return t.isoformat() if hasattr(t, "isoformat") else str(t)


def run_to_dict(br: BacktestRun) -> dict[str, Any]:
    res = br.result
    close = br.data["close"] if "close" in br.data.columns else br.data.iloc[:, 0]
    trades = trades_from_position(res.position, close)
    return {
        "model_id": br.model_id,
        "symbol": br.symbol,
        "params": _jsonify(br.params),
        "metrics": _jsonify(dict(res.metrics)),
        "config": _jsonify(dict(res.config)),
        "series": {
            "equity": _series(res.equity_curve),
            "returns": _series(res.returns),
            "position": _series(res.position),
        },
        "trades": [
            {
                "entry_time": _iso(r.entry_time),
                "exit_time": _iso(r.exit_time),
                "side": r.side,
                "entry_price": float(r.entry_price),
                "exit_price": float(r.exit_price),
                "pnl": float(r.pnl),
                "duration": int(r.duration),
            }
            for r in trades.itertuples(index=False)
        ],
    }
