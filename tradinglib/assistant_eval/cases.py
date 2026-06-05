"""Load held-out eval.jsonl gold traces into EvalCase objects.

Inverse of dataset/serialize.py: pull the user prompt, the gold tool calls, and
the gold final answer from each chat row. Malformed rows (per the trainer's own
validator) are skipped so a partial dataset still evaluates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tradinglib.assistant.types import ToolCall
from tradinglib.training.data import load_jsonl, validate_example


@dataclass(frozen=True)
class EvalCase:
    user_prompt: str
    gold_tool_calls: list[ToolCall]
    gold_answer: str
    category: str
    model_id: str


def _gold_calls(messages: list[dict]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                calls.append(
                    ToolCall(
                        id=tc.get("id", ""), name=fn["name"], input=json.loads(fn["arguments"])
                    )
                )
    return calls


def _to_case(row: dict) -> EvalCase:
    messages = row["messages"]
    user_prompt = next(m["content"] for m in messages if m.get("role") == "user")
    gold_answer = messages[-1].get("content", "")
    meta = row.get("meta", {})
    return EvalCase(
        user_prompt=user_prompt,
        gold_tool_calls=_gold_calls(messages),
        gold_answer=gold_answer,
        category=meta.get("category", ""),
        model_id=meta.get("model_id", ""),
    )


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    return [_to_case(row) for row in load_jsonl(path) if not validate_example(row)]
