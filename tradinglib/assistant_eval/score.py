"""Deterministic per-case scoring: tool-call accuracy and grounding.

Tool-call accuracy = greedy match of candidate calls to gold calls (same name +
args equal under numeric tolerance / order-independent keys / case-insensitive
strings), scored matched / max(|gold|, |candidate|) so both missing and spurious
calls cost. Grounding reuses the existing deterministic verifier unchanged."""

from __future__ import annotations

from typing import Any

from tradinglib.assistant.types import ToolCall
from tradinglib.assistant_eval.runner import RunResult
from tradinglib.dataset.grounding import is_grounded

_REL = 0.02
_ABS = 0.01


def _value_match(x: Any, y: Any) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return x is y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y) <= max(_ABS, _REL * abs(y))
    if isinstance(x, str) and isinstance(y, str):
        return x.strip().lower() == y.strip().lower()
    if isinstance(x, dict) and isinstance(y, dict):
        return _args_match(x, y)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_value_match(a, b) for a, b in zip(x, y, strict=False))
    return x == y


def _args_match(candidate: dict, gold: dict) -> bool:
    """The candidate must satisfy every arg the gold trace specified (right keys,
    matching values); extra optional args on the candidate are allowed. The
    minimal gold traces omit defaults the model often spells out (some are inert
    knobs the model doesn't even support), so penalizing that over-specification
    measured tool *selection* unfairly. Wrong or missing gold args still fail."""
    return all(k in candidate and _value_match(candidate[k], gold[k]) for k in gold)


def tool_call_score(candidate: list[ToolCall], gold: list[ToolCall]) -> float:
    if not gold:
        return 1.0 if not candidate else 0.0
    remaining = list(gold)
    matched = 0
    for c in candidate:
        for i, g in enumerate(remaining):
            if c.name == g.name and _args_match(c.input, g.input):
                matched += 1
                remaining.pop(i)
                break
    return matched / max(len(gold), len(candidate))


def grounded(run: RunResult) -> bool:
    ok, _missing = is_grounded(run.final_answer, run.tool_outputs)
    return ok
