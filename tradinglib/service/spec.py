# tradinglib/service/spec.py
"""Per-model capability descriptors built from model.md frontmatter.

``ModelSpec`` is the single source of truth for what a model accepts: its
ticker mode, tunable params, and whether it honors cost/sizing overrides.
Both the web GUI (renders widgets) and the LLM engine (knows legal knobs)
read it, so model-specific knowledge stays in the model, not the consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from tradinglib.models_index import find_models

TickerMode = Literal["free", "fixed", "choice"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    type: Literal["int", "float"]
    default: float
    min: float
    max: float


@dataclass(frozen=True)
class ModelSpec:
    id: str  # == frontmatter _path, e.g. "models/classical/01-sma-crossover-spy"
    name: str
    family: str
    assets: tuple[str, ...]
    ticker_mode: TickerMode
    ticker_choices: tuple[str, ...]
    default_ticker: str
    params: tuple[ParamSpec, ...]
    supports_costs: bool
    supports_sizing: bool

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)


def _ticker_mode(meta: dict[str, Any]) -> TickerMode:
    tickers = meta.get("tickers")
    if tickers == "any":
        return "free"
    if isinstance(tickers, (list, tuple)):
        return "fixed" if len(tickers) <= 1 else "choice"
    return "fixed"


def _default_ticker(meta: dict[str, Any], choices: tuple[str, ...]) -> str:
    explicit = meta.get("default_ticker")
    if explicit is not None:
        return str(explicit)
    if choices:
        return choices[0]
    raise ValueError(f"model {meta.get('name', '?')!r} has no default_ticker and no tickers list")


def _params(meta: dict[str, Any]) -> tuple[ParamSpec, ...]:
    raw = meta.get("params") or []
    return tuple(
        ParamSpec(
            name=p["name"],
            label=p.get("label", p["name"]),
            type=p["type"],
            default=p["default"],
            min=p["min"],
            max=p["max"],
        )
        for p in raw
    )


def _spec_from_meta(meta: dict[str, Any]) -> ModelSpec:
    choices = (
        tuple(str(t) for t in meta["tickers"])
        if isinstance(meta.get("tickers"), (list, tuple))
        else ()
    )
    return ModelSpec(
        id=meta["_path"],
        name=meta["name"],
        family=meta.get("family", ""),
        assets=tuple(meta.get("assets") or ()),
        ticker_mode=_ticker_mode(meta),
        ticker_choices=choices,
        default_ticker=_default_ticker(meta, choices),
        params=_params(meta),
        supports_costs=bool(meta.get("supports_costs", True)),
        supports_sizing=bool(meta.get("supports_sizing", True)),
    )


@lru_cache(maxsize=1)
def list_specs() -> tuple[ModelSpec, ...]:
    return tuple(_spec_from_meta(m) for m in find_models())


def model_spec(model_id: str) -> ModelSpec:
    for s in list_specs():
        if s.id == model_id:
            return s
    raise KeyError(model_id)
