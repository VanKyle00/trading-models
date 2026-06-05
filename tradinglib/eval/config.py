"""The ship-as-default gate: thresholds the candidate must clear to replace Claude.

Stricter grounding (>=0.99) is deliberate: a hallucinated metric is the worst
failure for a tool that reports live numbers. All three are CLI-overridable; the
regression axis (existing suite stays green) is enforced by CI, not this module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShipBar:
    tool_call_accuracy: float = 0.90
    grounded_fraction: float = 0.99
    judge_winrate: float = 0.45


DEFAULT_BAR = ShipBar()
