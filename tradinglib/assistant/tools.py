# tradinglib/assistant/tools.py
"""The assistant's tools: backtest service wrappers plus the options planner.

``dispatch`` NEVER raises on bad model input — validation/loader errors become a
(message, is_error=True) tuple the model self-corrects from. The tool schemas are
provider-neutral ({name, description, input_schema}); ClaudeProvider passes them
through unchanged.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from tradinglib.assistant import planner
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
    {
        "name": "propose_trade_levels",
        "description": (
            "Propose data-grounded trade levels for a hypothesis on a ticker. "
            "Directional stances (long/short): entry = last close (market order), "
            "stop = 2x ATR(14), target = 2R. Neutral stance (range-bound view): a "
            "lower/upper band = spot -/+ 2x ATR(14). Returns the levels plus "
            "spot/ATR context. Present them to the user for confirmation or "
            "adjustment before building a ticket."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Equity ticker, e.g. RIVN."},
                "stance": {"type": "string", "enum": ["long", "short", "neutral"]},
            },
            "required": ["ticker", "stance"],
        },
    },
    {
        "name": "build_options_ticket",
        "description": (
            "Build a sized, options-only trade ticket from user-confirmed levels "
            "using live option-chain quotes. Directional stances require "
            "entry/stop/target and return ranked structures (long option, debit "
            "spread, cash-secured put for longs, credit spread). The neutral stance "
            "requires lower/upper band bounds and returns defined-risk range "
            "structures (iron condor, iron butterfly); preference is ignored for "
            "neutral. Exactly one structure is recommended; every structure carries "
            "warnings and a prefilled profit-calculator link. Only call after the "
            "user confirmed levels, account size, and risk per trade."
            " Each structure carries a stable 'key'; rebuild with structure_key to "
            "pin a different candidate and recompute its plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "stance": {"type": "string", "enum": ["long", "short", "neutral"]},
                "entry": {"type": "number", "description": "Directional stances only."},
                "entry_type": {"type": "string", "enum": ["market", "stop", "limit"]},
                "stop": {"type": "number", "description": "Directional stances only."},
                "target": {"type": "number", "description": "Directional stances only."},
                "lower": {"type": "number", "description": "Neutral stance only: band floor."},
                "upper": {"type": "number", "description": "Neutral stance only: band ceiling."},
                "account_size": {"type": "number", "description": "Account size in dollars."},
                "risk_per_trade_pct": {
                    "type": "number",
                    "description": "Risk per trade as a FRACTION, e.g. 0.01 for 1%.",
                },
                "preference": {
                    "type": "string",
                    "enum": ["auto", "directional", "premium"],
                    "description": "Structure tilt; 'auto' lets the IV/RV ratio decide.",
                },
                "structure_key": {
                    "type": "string",
                    "description": (
                        "Pin the recommendation (and exit plan) to one candidate from the "
                        "most recent build_options_ticket result — pass that structure's "
                        "'key'. Omit on first build."
                    ),
                },
                "hypothesis": {"type": "string", "description": "The user's one-line thesis."},
            },
            "required": ["ticker", "stance", "account_size", "risk_per_trade_pct"],
        },
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


def _ticker_stance(args: dict[str, Any]) -> tuple[str, str] | str:
    """Common (ticker, stance) validation; returns an error string on failure."""
    ticker = str(args.get("ticker", "")).strip().upper()
    stance = str(args.get("stance", "")).strip().lower()
    if not ticker:
        return "ticker is required"
    if stance not in ("long", "short", "neutral"):
        return f"stance must be 'long', 'short', or 'neutral', got {stance!r}"
    return ticker, stance


def _propose_trade_levels(args: dict[str, Any]) -> tuple[str, bool]:
    parsed = _ticker_stance(args)
    if isinstance(parsed, str):
        return _err(parsed)
    ticker, stance = parsed
    try:
        return _ok(planner.propose_levels(ticker, stance))
    except Exception as exc:
        return _err(f"could not propose levels for {ticker}: {exc}")


def _build_options_ticket(args: dict[str, Any]) -> tuple[str, bool]:
    parsed = _ticker_stance(args)
    if isinstance(parsed, str):
        return _err(parsed)
    ticker, stance = parsed
    try:
        account_size = float(args["account_size"])
        risk = float(args["risk_per_trade_pct"])
    except (KeyError, TypeError, ValueError):
        return _err("account_size and risk_per_trade_pct are required numbers")
    if account_size <= 0:
        return _err("account_size must be positive")
    if not 0 < risk <= 0.2:
        return _err("risk_per_trade_pct must be a fraction (e.g. 0.01 for 1%), at most 0.2")
    preference = str(args.get("preference", "auto"))
    if preference not in ("auto", "directional", "premium"):
        return _err(f"preference must be auto|directional|premium, got {preference!r}")
    if stance == "neutral":
        try:
            lower = float(args["lower"])
            upper = float(args["upper"])
        except (KeyError, TypeError, ValueError):
            return _err("lower and upper are required numbers for a neutral stance")
        if not 0 < lower < upper:
            return _err("neutral geometry requires 0 < lower < upper")
        levels: dict[str, Any] = {"lower": lower, "upper": upper}
    else:
        try:
            entry = float(args["entry"])
            stop = float(args["stop"])
            target = float(args["target"])
        except (KeyError, TypeError, ValueError):
            return _err("entry, stop, and target are required numbers for a directional stance")
        entry_type = str(args.get("entry_type", "market"))
        if entry_type not in ("market", "stop", "limit"):
            return _err(f"entry_type must be market|stop|limit, got {entry_type!r}")
        if min(entry, stop, target) <= 0:
            return _err("entry, stop, and target must all be positive prices")
        if stance == "long" and not (stop < entry < target):
            return _err("long geometry requires stop < entry < target")
        if stance == "short" and not (target < entry < stop):
            return _err("short geometry requires target < entry < stop")
        levels = {"entry": entry, "entry_type": entry_type, "stop": stop, "target": target}
    structure_key = str(args["structure_key"]) if args.get("structure_key") else None
    try:
        ticket = planner.hypothesis_ticket(
            ticker=ticker,
            stance=stance,
            levels=levels,
            account_size=account_size,
            risk_per_trade_pct=risk,
            preference=preference,
            hypothesis=str(args.get("hypothesis", "")),
            structure_key=structure_key,
        )
        return _ok(ticket)
    except (
        ValueError
    ) as exc:  # validation: bad structure_key / level geometry — retrying won't help
        return _err(f"could not build a ticket for {ticker}: {exc}")
    except Exception as exc:
        return _err(
            f"could not build a ticket for {ticker}: {exc} — if this was a data fetch "
            "error, try again in a minute"
        )


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
        if name == "propose_trade_levels":
            return _propose_trade_levels(args)
        if name == "build_options_ticket":
            return _build_options_ticket(args)
        return _err(f"unknown tool {name!r}")
    except Exception as exc:  # tool boundary must never raise into the agent
        return _err(f"tool {name!r} failed: {type(exc).__name__}: {exc}")
