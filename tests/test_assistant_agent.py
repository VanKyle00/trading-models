# tests/test_assistant_agent.py
from tradinglib.assistant.agent import run_chat
from tradinglib.assistant.budget import Budget
from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, ToolCall, Usage


def _events(gen):
    return list(gen)


def test_plain_answer_no_tools():
    provider = StubProvider(
        [
            AssistantTurn(
                text="The SMA model is a baseline.",
                tool_calls=(),
                stop_reason="end_turn",
                usage=Usage(10, 8),
            ),
        ]
    )
    events = _events(run_chat("what is the sma model?", provider, Budget()))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "final"
    assert "baseline" in events[-1]["text"]
    assert kinds.count("text") == 0  # terminal answer goes only in 'final', not duplicated
    assert kinds.count("final") == 1


def test_runs_a_tool_then_answers():
    provider = StubProvider(
        [
            AssistantTurn(
                text="Looking it up.",
                tool_calls=(ToolCall("t1", "list_models", {}),),
                stop_reason="tool_use",
                usage=Usage(5, 5),
            ),
            AssistantTurn(
                text="There are five models.",
                tool_calls=(),
                stop_reason="end_turn",
                usage=Usage(5, 5),
            ),
        ]
    )
    events = _events(run_chat("how many models?", provider, Budget()))
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert kinds[-1] == "final"
    assert "five" in events[-1]["text"]


def test_budget_exhaustion_ends_gracefully():
    looping = [
        AssistantTurn(
            text="",
            tool_calls=(ToolCall(f"t{i}", "list_models", {}),),
            stop_reason="tool_use",
            usage=Usage(1, 1),
        )
        for i in range(5)
    ]
    looping_provider = StubProvider(looping)
    events = _events(
        run_chat("loop", looping_provider, Budget(max_tool_calls=1, max_tokens=10_000))
    )
    assert events[-1]["type"] == "final"
    assert "limit" in events[-1]["text"].lower()
    assert looping_provider.calls
