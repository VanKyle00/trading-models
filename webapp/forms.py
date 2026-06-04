# webapp/forms.py
"""Translate a flat request payload (HTML form fields or JSON) into a typed
BacktestRequest, coercing each declared param to its ModelSpec type.

Parsing/coercion errors surface as RequestError so the route layer maps them to
HTTP 400 uniformly with validate()'s own RequestError.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from tradinglib.service import BacktestRequest, ModelSpec, RequestError


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise RequestError(f"invalid {field} date {value!r}: {exc}") from exc


def _opt_float(payload: dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError) as exc:
        raise RequestError(f"{key} must be a number, got {raw!r}") from exc


def request_from_payload(payload: dict[str, Any], spec: ModelSpec) -> BacktestRequest:
    params: dict[str, Any] = {}
    for p in spec.params:
        raw = payload.get(p.name)
        if raw is None or str(raw).strip() == "":
            continue
        cast = int if p.type == "int" else float
        try:
            params[p.name] = cast(raw)
        except (ValueError, TypeError) as exc:
            raise RequestError(f"param {p.name!r} must be {p.type}, got {raw!r}") from exc

    raw_symbol = payload.get("symbol")
    if isinstance(raw_symbol, (list, tuple)):
        raw_symbol = raw_symbol[0] if raw_symbol else None
    symbol = str(raw_symbol).strip() if raw_symbol is not None else None
    symbol = symbol or None  # blank → None

    return BacktestRequest(
        model_id=str(payload["model_id"]),
        start=_parse_date(str(payload.get("start", "")), "start"),
        end=_parse_date(str(payload.get("end", "")), "end"),
        symbol=symbol,
        fee_bps=_opt_float(payload, "fee_bps"),
        slippage_bps=_opt_float(payload, "slippage_bps"),
        initial_capital=_opt_float(payload, "initial_capital"),
        size_mult=_opt_float(payload, "size_mult"),
        params=params,
    )
