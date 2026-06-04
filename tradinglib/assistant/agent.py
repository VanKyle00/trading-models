# tradinglib/assistant/agent.py
"""The bounded tool-use loop.

``run_chat`` drives a provider through propose-tool → dispatch → feed-result until
the model gives a final answer or the budget is exhausted. It yields plain dict
events the SSE route serializes. Provider-neutral: it never imports anthropic.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from tradinglib.assistant.budget import Budget, BudgetExceeded
from tradinglib.assistant.provider import SYSTEM_PROMPT, LLMProvider
from tradinglib.assistant.tools import TOOL_SPECS, dispatch
from tradinglib.assistant.types import (
    AssistantMsg,
    Message,
    ToolResult,
    ToolResultMsg,
    UserMsg,
)

_MAX_TURNS = 16  # hard backstop on top of the budget


def run_chat(user_message: str, provider: LLMProvider, budget: Budget) -> Iterator[dict[str, Any]]:
    conversation: list[Message] = [UserMsg(user_message)]

    for _ in range(_MAX_TURNS):
        try:
            turn = provider.complete(SYSTEM_PROMPT, conversation, TOOL_SPECS)
            budget.charge_tokens(turn.usage.total)
        except BudgetExceeded:
            yield {"type": "final", "text": "Session limit reached — start a new chat to continue."}
            return

        if turn.stop_reason != "tool_use" or not turn.tool_calls:
            yield {"type": "final", "text": turn.text}
            return

        # interim narration the model wrote before its tool calls
        if turn.text:
            yield {"type": "text", "text": turn.text}

        conversation.append(AssistantMsg(turn))

        results: list[ToolResult] = []
        for call in turn.tool_calls:
            try:
                budget.charge_tool_call()
            except BudgetExceeded:
                yield {"type": "final", "text": "Reached the per-session backtest limit."}
                return
            yield {"type": "tool_call", "name": call.name, "input": call.input}
            content, is_error = dispatch(call.name, call.input)
            yield {"type": "tool_result", "name": call.name, "is_error": is_error}
            results.append(ToolResult(call.id, content, is_error))

        conversation.append(ToolResultMsg(tuple(results)))

    yield {"type": "final", "text": "Stopped after too many steps. Please refine your question."}
