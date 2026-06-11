# tradinglib/assistant/agent.py
"""The bounded tool-use loop.

``run_chat`` drives a provider through propose-tool → dispatch → feed-result until
the model gives a final answer or the budget is exhausted. It yields plain dict
events the SSE route serializes. Provider-neutral: it never imports anthropic.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from tradinglib.assistant.budget import Budget, BudgetExceeded
from tradinglib.assistant.provider import SYSTEM_PROMPT, LLMProvider
from tradinglib.assistant.tools import TOOL_SPECS, dispatch
from tradinglib.assistant.types import (
    AssistantMsg,
    AssistantTurn,
    Message,
    ToolResult,
    ToolResultMsg,
    Usage,
    UserMsg,
)

_MAX_TURNS = 16  # hard backstop on top of the budget


def _seed(history: Sequence[tuple[str, str]] | None, opening: str) -> list[Message]:
    """History pairs -> alternating Message list ending with the opening UserMsg.

    Same-role runs are merged and leading assistant entries dropped: providers
    reject conversations that don't alternate or don't start with a user turn.
    Replayed assistant turns are plain text (tool calls are not reconstructed).
    """
    out: list[Message] = []
    user_buf: list[str] = []

    def flush_user() -> None:
        if user_buf:
            out.append(UserMsg("\n\n".join(user_buf)))
            user_buf.clear()

    for role, text in history or ():
        if not text:
            continue
        if role == "user":
            user_buf.append(text)
        else:
            if not out and not user_buf:
                continue  # the first turn must be a user turn
            flush_user()
            last = out[-1]
            if isinstance(last, AssistantMsg):
                merged = AssistantTurn(
                    last.turn.text + "\n\n" + text, (), "end_turn", last.turn.usage
                )
                out[-1] = AssistantMsg(merged)
            else:
                out.append(AssistantMsg(AssistantTurn(text, (), "end_turn", Usage(0, 0))))
    user_buf.append(opening)
    flush_user()
    return out


def run_chat(
    user_message: str,
    provider: LLMProvider,
    budget: Budget,
    context: str | None = None,
    history: Sequence[tuple[str, str]] | None = None,
    settings: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Drive the bounded tool-use loop, yielding SSE event dicts.

    ``context`` (optional) is a plain-text summary of the backtest the user is
    currently looking at. When present it is folded into the opening message so
    the assistant can answer "explain the worst month" against the on-screen run
    without first re-running it. Absent, the assistant behaves as a standalone
    agent that runs its own backtests via its tools.

    ``history`` (optional) is the client-replayed transcript — (role, text)
    pairs of prior user messages and final assistant replies. The server holds
    no session state; the browser is the source of truth for the conversation.

    ``settings`` (optional) is the /planner sizing line (account size, risk per
    trade) the webapp renders from the page's settings strip. When present it
    is appended to the opening message with an instruction to use it for
    sizing and never ask.
    """
    opening = user_message
    if context:
        opening = (
            "The user is looking at this backtest result on screen:\n"
            f"{context}\n\n"
            f"Their question: {user_message}"
        )
    if settings:
        opening = (
            f"{opening}\n\n{settings}\n"
            "Use these for sizing; do not ask the user for account size or risk."
        )
    conversation: list[Message] = _seed(history, opening)

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
            yield {
                "type": "tool_result",
                "name": call.name,
                "is_error": is_error,
                "output": content,
            }
            results.append(ToolResult(call.id, content, is_error))

        conversation.append(ToolResultMsg(tuple(results)))

    yield {"type": "final", "text": "Stopped after too many steps. Please refine your question."}
