# tradinglib/assistant/tools.py
"""The three tools the agent may call, each a thin wrapper over tradinglib.service.

``dispatch`` NEVER raises on bad model input — validation/loader errors become a
(message, is_error=True) tuple the model self-corrects from. The tool schemas are
provider-neutral ({name, description, input_schema}); ClaudeProvider passes them
through unchanged.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from tradinglib.service import (
    BacktestRequest,
    RequestError,
    list_specs,
    model_spec,
    run,
    run_to_dict,
)

_PARAM_SCHEMA = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string", "description": "Model id from list_models."},
        "symbol": {"type": "string", "description": "Ticker (omit to use the model default)."},
        "start": {"type": "string", "description": "ISO date YYYY-MM-DD."},
        "end": {"type": "string", "description": "ISO date YYYY-MM-DD."},
        "fee_bps": {"type": "number"},
        "slippage_bps": {"type": "number"},
        "initial_capital": {"type": "number"},
        "size_mult": {"type": "number"},
        "params": {
            "type": "object",
            "description": "Model-specific strategy params (see get_model_spec).",
        },
    },
    "required": ["model_id", "start", "end"],
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_models",
        "description": "List the available backtest models (id, name, family, assets).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_model_spec",
        "description": (
            "Get a model's allowed tickers, tunable params (name/type/min/max), and "
            "capability flags. Call before run_backtest to learn the legal knobs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"}},
            "required": ["model_id"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Run a backtest and get its metrics. Vary symbol/dates/params to run "
            "counterfactuals. Returns metrics + summary stats (not full series)."
        ),
        "input_schema": _PARAM_SCHEMA,
    },
]


def _ok(payload: dict[str, Any]) -> tuple[str, bool]:
    return json.dumps(payload), False


def _err(message: str) -> tuple[str, bool]:
    return message, True


def _list_models() -> tuple[str, bool]:
    models = [
        {"id": s.id, "name": s.name, "family": s.family, "assets": list(s.assets)}
        for s in list_specs()
    ]
    return _ok({"models": models})


def _get_model_spec(args: dict[str, Any]) -> tuple[str, bool]:
    try:
        spec = model_spec(str(args.get("model_id", "")))
    except KeyError as exc:
        return _err(f"unknown model_id {exc}")
    return _ok(
        {
            "id": spec.id,
            "name": spec.name,
            "ticker_mode": spec.ticker_mode,
            "ticker_choices": list(spec.ticker_choices),
            "default_ticker": spec.default_ticker,
            "supports_costs": spec.supports_costs,
            "supports_sizing": spec.supports_sizing,
            "params": [
                {"name": p.name, "type": p.type, "default": p.default, "min": p.min, "max": p.max}
                for p in spec.params
            ],
        }
    )


def _run_backtest(args: dict[str, Any]) -> tuple[str, bool]:
    try:
        req = BacktestRequest(
            model_id=str(args["model_id"]),
            start=date.fromisoformat(str(args["start"])),
            end=date.fromisoformat(str(args["end"])),
            symbol=args.get("symbol"),
            fee_bps=args.get("fee_bps"),
            slippage_bps=args.get("slippage_bps"),
            initial_capital=args.get("initial_capital"),
            size_mult=args.get("size_mult"),
            params=args.get("params") or {},
        )
        result = run_to_dict(run(req))
        equity = result["series"]["equity"]["values"]
        return _ok(
            {
                "model_id": result["model_id"],
                "symbol": result["symbol"],
                "params": result["params"],
                # Effective run config (initial_capital, fees, slippage) so the
                # assistant can ground references to starting capital / costs.
                "config": result["config"],
                "metrics": result["metrics"],
                "n_trades": len(result["trades"]),
                "final_equity": equity[-1] if equity else None,
            }
        )
    except (RequestError, KeyError, ValueError) as exc:
        return _err(f"could not run: {exc}")


def dispatch(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool. Guarantees it never raises — any failure becomes (msg, is_error)
    so the agent loop and the public SSE stream can't be broken by a tool error."""
    try:
        if name == "list_models":
            return _list_models()
        if name == "get_model_spec":
            return _get_model_spec(args)
        if name == "run_backtest":
            return _run_backtest(args)
        return _err(f"unknown tool {name!r}")
    except Exception as exc:  # tool boundary must never raise into the agent
        return _err(f"tool {name!r} failed: {type(exc).__name__}: {exc}")
