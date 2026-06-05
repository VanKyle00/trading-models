"""Serialize a RecordedTrace to an OpenAI/Qwen-style chat training example.

Output shape (one JSON object per line in the final JSONL):
{"messages": [
   {"role": "system", "content": ...},
   {"role": "user", "content": ...},
   {"role": "assistant", "content": "", "tool_calls": [{"id","type":"function",
        "function": {"name", "arguments": "<json str>"}}]},
   {"role": "tool", "tool_call_id": ..., "content": ...},
   {"role": "assistant", "content": "<final answer>"}],
 "meta": {"category", "model_id"}}
This is what TRL/axolotl/unsloth SFT trainers consume for tool-use fine-tuning."""

from __future__ import annotations

import json
from typing import Any

from tradinglib.assistant.provider import SYSTEM_PROMPT
from tradinglib.assistant.types import AssistantMsg, ToolResultMsg, UserMsg
from tradinglib.dataset.recorder import RecordedTrace


def to_chat_example(trace: RecordedTrace, system: str = SYSTEM_PROMPT) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in trace.conversation:
        if isinstance(msg, UserMsg):
            messages.append({"role": "user", "content": msg.text})
        elif isinstance(msg, AssistantMsg):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.turn.text}
            if msg.turn.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                    }
                    for tc in msg.turn.tool_calls
                ]
            messages.append(entry)
        elif isinstance(msg, ToolResultMsg):
            for r in msg.results:
                messages.append(
                    {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
                )
    messages.append({"role": "assistant", "content": trace.final_answer})
    return {
        "messages": messages,
        "meta": {"category": trace.scenario.category, "model_id": trace.scenario.model_id},
    }
