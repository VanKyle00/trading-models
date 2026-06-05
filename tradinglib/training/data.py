"""Load and validate the harness JSONL, and shape it for the SFT trainer.

Pure stdlib - imports NO heavy ML deps, so it runs in the normal test env.
The trainer (scripts/train_assistant.py) consumes ``to_trl_records`` output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROLES = {"system", "user", "assistant", "tool"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_example(ex: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty == valid)."""
    problems: list[str] = []
    messages = ex.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["missing or empty 'messages'"]

    open_tool_call = False
    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in _ROLES:
            problems.append(f"message {i}: invalid role {role!r}")
            continue
        if role == "assistant":
            open_tool_call = bool(m.get("tool_calls"))
        elif role == "tool":
            if not open_tool_call:
                problems.append(f"message {i}: tool result without a preceding assistant tool_call")
        else:
            open_tool_call = False

    if messages[-1].get("role") != "assistant":
        problems.append("last message must be an assistant answer")
    return problems


def validate_dataset(path: str | Path) -> tuple[int, list[str]]:
    """Return (n_valid, problems) where problems are one entry per bad example."""
    n_ok = 0
    problems: list[str] = []
    for idx, ex in enumerate(load_jsonl(path)):
        errs = validate_example(ex)
        if errs:
            problems.append(f"line {idx}: " + "; ".join(errs))
        else:
            n_ok += 1
    return n_ok, problems


def to_trl_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip everything but the conversational ``messages`` the trainer needs."""
    return [{"messages": ex["messages"]} for ex in examples]
