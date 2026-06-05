# tradinglib/assistant/provider.py
"""LLM provider abstraction.

The agent loop depends only on the ``LLMProvider`` protocol and the neutral
types. ``ClaudeProvider`` is the default (anthropic SDK, Haiku 4.5). A future
self-hosted/own-trained model implements ``LLMProvider`` and drops in with no
changes to agent.py or tools.py. This is the ONLY module that imports anthropic.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from tradinglib.assistant.types import (
    AssistantMsg,
    AssistantTurn,
    Message,
    ToolCall,
    ToolResultMsg,
    Usage,
    UserMsg,
)

DEFAULT_MODEL = os.environ.get("ASSISTANT_MODEL", "claude-haiku-4-5")
SYSTEM_PROMPT = (
    "You are a quantitative-research assistant for THIS portfolio of backtest "
    "models. Answer questions about the models and their results. You can run "
    "counterfactual backtests by calling run_backtest with varied tickers, dates, "
    "or params. ALWAYS ground numeric claims in metrics returned by your tools — "
    "never invent numbers. State only numbers that appear in a tool result "
    "(including the run config the tools report, e.g. initial capital and fees); "
    "do not compute or estimate figures a tool did not return — annualized returns, "
    "percentages of capital, or train/test-split conventions — and report metric "
    "signs exactly as returned. Call get_model_spec before run_backtest to learn a "
    "model's legal knobs. Be concise."
)


class LLMProvider(Protocol):
    def complete(
        self, system: str, conversation: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantTurn: ...


def _to_anthropic_messages(conversation: list[Message]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for m in conversation:
        if isinstance(m, UserMsg):
            msgs.append({"role": "user", "content": m.text})
        elif isinstance(m, AssistantMsg):
            content: list[dict[str, Any]] = []
            if m.turn.text:
                content.append({"type": "text", "text": m.turn.text})
            for tc in m.turn.tool_calls:
                content.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                )
            msgs.append({"role": "assistant", "content": content})
        elif isinstance(m, ToolResultMsg):
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.tool_call_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                        for r in m.results
                    ],
                }
            )
    return msgs


def _turn_from_response(resp: Any) -> AssistantTurn:
    text = "".join(b.text for b in resp.content if b.type == "text")
    tool_calls = tuple(
        ToolCall(id=b.id, name=b.name, input=dict(b.input))
        for b in resp.content
        if b.type == "tool_use"
    )
    return AssistantTurn(
        text=text,
        tool_calls=tool_calls,
        stop_reason=resp.stop_reason or "end_turn",
        usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
    )


class ClaudeProvider:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048) -> None:
        import anthropic  # lazy: tests/non-LLM paths don't need the SDK or a key

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete(
        self, system: str, conversation: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,  # type: ignore[arg-type]  # plain dicts; SDK accepts at runtime
            messages=_to_anthropic_messages(conversation),  # type: ignore[arg-type]
        )
        return _turn_from_response(resp)


class StubProvider:
    """Deterministic provider for tests — yields scripted turns in order."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[list[Message]] = []

    def complete(
        self, system: str, conversation: list[Message], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        self.calls.append(list(conversation))
        if not self._turns:
            raise AssertionError("StubProvider ran out of scripted turns")
        return self._turns.pop(0)
