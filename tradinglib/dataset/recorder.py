"""Drive a provider through the real tools and capture the full trace.

Mirrors the bounded loop in `tradinglib.assistant.agent` but RETURNS the
structured conversation + raw tool outputs (the agent loop only yields SSE
events and discards them). Parameterized by (tool_specs, dispatch) so it can use
the dataset tool set without touching runtime code."""

from __future__ import annotations

from dataclasses import dataclass, field

from tradinglib.assistant.budget import Budget, BudgetExceeded
from tradinglib.assistant.provider import SYSTEM_PROMPT, LLMProvider
from tradinglib.assistant.types import (
    AssistantMsg,
    Message,
    ToolResult,
    ToolResultMsg,
    UserMsg,
)
from tradinglib.dataset.scenarios import Scenario
from tradinglib.dataset.tools import DispatchFn

_MAX_TURNS = 16


@dataclass
class RecordedTrace:
    scenario: Scenario
    conversation: list[Message]
    tool_outputs: list[str] = field(default_factory=list)
    final_answer: str = ""
    ok: bool = False
    error: str = ""  # provider/infra error (e.g. API credit/rate-limit), "" if none


def record_trace(
    scenario: Scenario,
    provider: LLMProvider,
    budget: Budget,
    tool_specs: list[dict],
    dispatch: DispatchFn,
) -> RecordedTrace:
    conversation: list[Message] = [UserMsg(scenario.question)]
    tool_outputs: list[str] = []
    last_error = ""

    for _ in range(_MAX_TURNS):
        try:
            turn = provider.complete(SYSTEM_PROMPT, conversation, tool_specs)
            budget.charge_tokens(turn.usage.total)
        except BudgetExceeded:
            break
        except Exception as exc:
            # Capture (don't swallow) infra errors so the builder can tell an
            # API outage (credits/rate-limit/key) from a model that just didn't
            # answer, and abort instead of writing a silently-broken dataset.
            last_error = f"{type(exc).__name__}: {exc}"
            break

        if turn.stop_reason != "tool_use" or not turn.tool_calls:
            return RecordedTrace(
                scenario=scenario,
                conversation=conversation,
                tool_outputs=tool_outputs,
                final_answer=turn.text,
                ok=bool(turn.text.strip()),
            )

        conversation.append(AssistantMsg(turn))
        results: list[ToolResult] = []
        for call in turn.tool_calls:
            try:
                budget.charge_tool_call()
            except BudgetExceeded:
                return RecordedTrace(scenario, conversation, tool_outputs, "", False)
            content, is_error = dispatch(call.name, call.input)
            if not is_error:
                tool_outputs.append(content)
            results.append(ToolResult(call.id, content, is_error))
        conversation.append(ToolResultMsg(tuple(results)))

    return RecordedTrace(scenario, conversation, tool_outputs, "", False, error=last_error)
