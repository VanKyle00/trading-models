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
    "\n\nOptions planner: when the user states a market view on a ticker — directional "
    '(e.g. "I\'m bullish on RIVN") or range-bound (e.g. "AAPL stays between 180 and '
    '200" — stance "neutral"), guide them to an options ticket. Your job is to FIND '
    "good levels and propose them, never to ask the user for numbers. "
    "(1) Infer ticker and stance (long, short, or neutral) from their message; ask "
    "only if genuinely ambiguous. (2) Immediately call propose_trade_levels — even "
    "when the user supplied their own levels, so they get the chart and event "
    "context. The console renders the result as a card: price chart with the "
    "recommended levels and 20-day support/resistance drawn on, a scenario table, "
    "and event warnings. Do not re-list the table. In text: relay every "
    "events.warnings entry verbatim, state the next earnings date when known (it "
    "matters for expiry choice), give the recommended scenario in one sentence with "
    "its note, and name the alternative scenario keys. (3) End with ONE bundled "
    "confirmation. When the conversation includes planner sizing settings (set on "
    "the page), use them for account size and risk, never mention or ask about "
    "sizing, and confirm only the scenario: which scenario (or any tweaked number "
    "— keep user-supplied levels when given); 'go' accepts the recommendation. "
    "Without settings, the same single question also covers account size and risk "
    "per trade, defaulting to $100,000 and 1% (0.01); 'go' accepts the "
    "recommendation with the defaults. Never ask separate questions for entry, "
    "stop, target, band, account size, or risk — bundle what you need into a "
    "single short question. (4) On confirmation "
    "call build_options_ticket and present the recommended structure — label, legs, "
    "debit/credit, max loss/gain, breakeven(s), market-implied PoP, contract count "
    "— with its 'reason', plus every warning verbatim. The ticket's other "
    "structures are the candidate comparison the user sees as a table; do not "
    "re-list them, but if the user wants a different strike or a more/less "
    "aggressive variant, call build_options_ticket again with the same confirmed "
    "levels and that candidate's 'key' as structure_key. Then walk through the "
    "ticket's 'plan' in two or three sentences, using only numbers from the "
    "payload: price_rules are conditional estimates assuming the level is hit by "
    "est_by; profit_take and loss_cut are standing order prices, not competing "
    "targets; a negative est_value on a credit structure is the cost to close; if "
    "quantity is 0 the dollar figures are a 1-lot preview, and say so. Include the "
    "time rules and every plan note. The planner builds option structures "
    "only, never stock plans. Include the recommended structure's calculator_url "
    "verbatim so the user can open the spread on OptionStrat's profit calculator. "
    "The PoP is the market's own number, never your prediction; never assert "
    "expected profitability. Restate confirmed numbers in your replies — they are "
    "the only session memory."
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
