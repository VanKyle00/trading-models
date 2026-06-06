"""Deterministic numeric grounding check.

A trace is grounded if every numeric *claim* in the final answer matches a number
present in a real tool output (within tolerance). Reused at data-build time here;
the same logic is intended for runtime in sub-project 4.
"""

from __future__ import annotations

import re

# A leading '-' counts as a sign only at a token boundary, not when it follows a
# digit/dot -- otherwise hyphenated ranges and ISO dates ("2020-2024",
# "2023-12-31") spuriously parse the trailing field as negative.
_NUM = re.compile(r"(?<![\d.])-?\$?\d[\d,]*(?:\.\d+)?%?")
_IGNORE_BELOW = 10.0  # bare small ints are usually prose, not metric claims
# LLMs routinely emit a typographic minus or dash in tables (not ASCII "-"); fold
# them to ASCII '-' so a correctly-signed claim isn't misread as positive.
_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-"})  # noqa: RUF001
# Numbers inside index/proper-noun names ("S&P 500", "Russell 2000") are names,
# not metric claims -- strip the names before extraction so they aren't flagged
# ungrounded (also fixes false drops at dataset-build time).
_INDEX_NAMES = re.compile(
    r"S&P\s*500|S&P\s*600|Russell\s*2000|Russell\s*1000|Nasdaq[- ]?100|FTSE\s*100|Dow\s*30",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> set[float]:
    out: set[float] = set()
    text = _INDEX_NAMES.sub(" ", text.translate(_DASHES))
    for tok in _NUM.findall(text):
        is_pct = tok.endswith("%")
        cleaned = tok.replace("$", "").replace(",", "").rstrip("%")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        out.add(val / 100.0 if is_pct else val)
    return out


def _matches(claim: float, pool: set[float], rel: float = 0.05, abs_: float = 0.01) -> bool:
    return any(abs(claim - p) <= max(abs_, rel * abs(p)) for p in pool)


def is_grounded(
    answer: str, tool_outputs: list[str], rel: float = 0.05
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
