"""Scenario generation: model x legal ticker x question template."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    category: str
    question: str
    model_id: str  # "" for model-agnostic refusals
    symbol: str | None  # None -> tool uses model default
    start: str
    end: str
    params: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)  # e.g. {"fee_bps": 10}
