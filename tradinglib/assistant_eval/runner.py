"""Drive a candidate provider through the real tools and capture what it did.

Mirrors dataset/recorder.py:record_trace, but returns a flat tool_calls list (for
tool-call scoring) and the raw non-error tool outputs (for grounding). The shipped
agent loop is NOT reused: its SSE tool_result event hides the output content the
grounding check needs, and we will not change that public contract. Parameterized
by (tool_specs, dispatch) so tests inject a fake dispatch instead of running a real
backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

from tradinglib.assistant.budget import BudgetExceeded
from tradinglib.assistant.provider import SYSTEM_PROMPT
from tradinglib.assistant.types import (
    AssistantMsg,
    Message,
    ToolCall,
    ToolResult,
    ToolResultMsg,
    UserMsg,
)

_MAX_TURNS = 16


@dataclass
class RunResult:
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    final_answer: str = ""
    stopped: str = "answered"  # "answered" | "budget" | "max_turns"


def run_candidate(user_prompt, provider, budget, tool_specs, dispatch) -> RunResult:
    conversation: list[Message] = [UserMsg(user_prompt)]
    result = RunResult()

    for _ in range(_MAX_TURNS):
        try:
            turn = provider.complete(SYSTEM_PROMPT, conversation, tool_specs)
            budget.charge_tokens(turn.usage.total)
        except BudgetExceeded:
            result.stopped = "budget"
            return result

        if turn.stop_reason != "tool_use" or not turn.tool_calls:
            result.final_answer = turn.text
            result.stopped = "answered"
            return result

        conversation.append(AssistantMsg(turn))
        results: list[ToolResult] = []
        for call in turn.tool_calls:
            try:
                budget.charge_tool_call()
            except BudgetExceeded:
                result.stopped = "budget"
                return result
            result.tool_calls.append(call)
            content, is_error = dispatch(call.name, call.input)
            if not is_error:
                result.tool_outputs.append(content)
            results.append(ToolResult(call.id, content, is_error))
        conversation.append(ToolResultMsg(tuple(results)))

    result.stopped = "max_turns"
    return result
