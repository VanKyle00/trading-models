"""Deterministic numeric grounding check.

A trace is grounded if every numeric *claim* in the final answer matches a number
present in a real tool output (within tolerance). Reused at data-build time here;
the same logic is intended for runtime in sub-project 4.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
_IGNORE_BELOW = 10.0  # bare small ints are usually prose, not metric claims


def extract_numbers(text: str) -> set[float]:
    out: set[float] = set()
    for tok in _NUM.findall(text):
        is_pct = tok.endswith("%")
        cleaned = tok.replace("$", "").replace(",", "").rstrip("%")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        out.add(val / 100.0 if is_pct else val)
    return out


def _matches(claim: float, pool: set[float], rel: float = 0.02, abs_: float = 0.01) -> bool:
    return any(abs(claim - p) <= max(abs_, rel * abs(p)) for p in pool)


def is_grounded(
    answer: str, tool_outputs: list[str], rel: float = 0.02
) -> tuple[bool, list[float]]:
    """Return (ok, missing). ``missing`` lists claimed numbers not found in any
    tool output. Bare integers below ``_IGNORE_BELOW`` are treated as prose."""
    pool: set[float] = set()
    for out in tool_outputs:
        pool |= extract_numbers(out)
    missing = [
        c
        for c in extract_numbers(answer)
        if not (c.is_integer() and abs(c) < _IGNORE_BELOW) and not _matches(c, pool, rel)
    ]
    return (not missing, missing)
