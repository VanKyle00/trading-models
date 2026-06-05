import json

from tradinglib.assistant.types import (
    AssistantMsg,
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultMsg,
    Usage,
    UserMsg,
)
from tradinglib.dataset.recorder import RecordedTrace
from tradinglib.dataset.scenarios import Scenario
from tradinglib.dataset.serialize import to_chat_example


def _trace():
    turn = AssistantTurn(
        text="",
        stop_reason="tool_use",
        tool_calls=(
            ToolCall(
                id="t1", name="run_backtest", input={"model_id": "m", "start": "a", "end": "b"}
            ),
        ),
        usage=Usage(1, 1),
    )
    conv = [
        UserMsg("How did model m do?"),
        AssistantMsg(turn),
        ToolResultMsg((ToolResult("t1", '{"metrics": {"sharpe": 0.96}}', False),)),
    ]
    scn = Scenario(
        category="explain",
        question="How did model m do?",
        model_id="m",
        symbol="SPY",
        start="a",
        end="b",
    )
    return RecordedTrace(scn, conv, ['{"metrics": {"sharpe": 0.96}}'], "Sharpe was 0.96.", ok=True)


def test_messages_roles_in_order():
    ex = to_chat_example(_trace())
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_tool_call_arguments_are_json_string():
    ex = to_chat_example(_trace())
    call = ex["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == "run_backtest"
    assert json.loads(call["function"]["arguments"])["model_id"] == "m"


def test_final_assistant_message_is_answer():
    ex = to_chat_example(_trace())
    assert ex["messages"][-1] == {"role": "assistant", "content": "Sharpe was 0.96."}


def test_meta_carries_category_and_model():
    ex = to_chat_example(_trace())
    assert ex["meta"]["category"] == "explain"
    assert ex["meta"]["model_id"] == "m"
