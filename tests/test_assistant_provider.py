# tests/test_assistant_provider.py
from tradinglib.assistant.provider import StubProvider, _to_anthropic_messages
from tradinglib.assistant.types import (
    AssistantMsg,
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultMsg,
    Usage,
    UserMsg,
)


def test_stub_provider_returns_scripted_turns():
    turn = AssistantTurn(text="hi", tool_calls=(), stop_reason="end_turn", usage=Usage(10, 5))
    p = StubProvider([turn])
    out = p.complete("sys", [UserMsg("hello")], tools=[])
    assert out is turn


def test_to_anthropic_messages_roundtrips_tool_cycle():
    convo = [
        UserMsg("compare SPY and QQQ"),
        AssistantMsg(
            AssistantTurn(
                text="running",
                tool_calls=(ToolCall("t1", "run_backtest", {"x": 1}),),
                stop_reason="tool_use",
                usage=Usage(0, 0),
            )
        ),
        ToolResultMsg((ToolResult("t1", '{"sharpe": 0.5}', is_error=False),)),
    ]
    msgs = _to_anthropic_messages(convo)
    assert msgs[0] == {"role": "user", "content": "compare SPY and QQQ"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["type"] == "text"
    assert msgs[1]["content"][-1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "run_backtest",
        "input": {"x": 1},
    }
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": '{"sharpe": 0.5}',
        "is_error": False,
    }


def test_empty_assistant_text_omits_text_block():
    convo = [
        UserMsg("go"),
        AssistantMsg(
            AssistantTurn(
                text="",
                tool_calls=(ToolCall("t1", "list_models", {}),),
                stop_reason="tool_use",
                usage=Usage(0, 0),
            )
        ),
    ]
    msgs = _to_anthropic_messages(convo)
    assert all(b["type"] != "text" for b in msgs[1]["content"])
