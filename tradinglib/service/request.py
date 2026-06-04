# tradinglib/service/request.py
"""Typed backtest request + shared validation.

A ``BacktestRequest`` is what every consumer (web GUI, LLM engine) builds and
hands to :func:`tradinglib.service.run`. ``validate`` is the single gate: it is
reused by the web layer (→ HTTP 400) and the LLM tool layer (→ tool error the
model self-corrects from).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tradinglib.service.spec import ModelSpec


class RequestError(ValueError):
    """Raised when a BacktestRequest fails validation against its ModelSpec."""


@dataclass(frozen=True)
class BacktestRequest:
    model_id: str
    start: date
    end: date
    symbol: str | None = None  # None → ModelSpec.default_ticker
    fee_bps: float | None = None  # None → model's own default
    slippage_bps: float | None = None
    initial_capital: float | None = None
    size_mult: float | None = None
    params: dict[str, Any] = field(default_factory=dict)


def validate(request: BacktestRequest, spec: ModelSpec) -> None:
    """Raise :class:`RequestError` if ``request`` is not runnable against ``spec``."""
    if request.start >= request.end:
        raise RequestError(f"start ({request.start}) must be before end ({request.end})")

    _validate_symbol(request.symbol, spec)
    _validate_params(request.params, spec)

    if (
        request.fee_bps is not None or request.slippage_bps is not None
    ) and not spec.supports_costs:
        raise RequestError(f"model {spec.name!r} does not support cost overrides")
    if (
        request.size_mult is not None or request.initial_capital is not None
    ) and not spec.supports_sizing:
        raise RequestError(
            f"model {spec.name!r} does not support position sizing or capital overrides"
        )


def _validate_symbol(symbol: str | None, spec: ModelSpec) -> None:
    if symbol is None:
        return
    sym = symbol.strip().upper()
    if not sym:
        raise RequestError("symbol must be non-empty")
    if spec.ticker_mode == "free":
        return
    if spec.ticker_mode == "choice":
        choices = {c.upper() for c in spec.ticker_choices}
        if sym not in choices:
            raise RequestError(f"{sym!r} not in allowed tickers {sorted(choices)}")
    else:  # fixed — locked
        if sym != spec.default_ticker.upper():
            raise RequestError(f"model {spec.name!r} is locked to {spec.default_ticker}")


def _validate_params(params: dict[str, Any], spec: ModelSpec) -> None:
    for name, value in params.items():
        p = spec.param(name)
        if p is None:
            raise RequestError(f"unknown param {name!r} for model {spec.name!r}")
        if p.type == "int" and not (isinstance(value, int) and not isinstance(value, bool)):
            raise RequestError(f"param {name!r} must be type int, got {type(value).__name__}")
        if p.type == "float" and not (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            raise RequestError(f"param {name!r} must be type float, got {type(value).__name__}")
        if not (p.min <= value <= p.max):
            raise RequestError(f"param {name!r}={value} out of range [{p.min}, {p.max}]")
