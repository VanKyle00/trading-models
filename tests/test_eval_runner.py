from tradinglib.assistant.budget import Budget
from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, ToolCall, Usage
from tradinglib.assistant_eval.runner import RunResult, run_candidate

_TOOLS = [{"name": "run_backtest", "description": "", "input_schema": {"type": "object"}}]


def _fake_dispatch(name, args):
    return '{"metrics": {"sharpe": 1.5}}', False


def _final(text):
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(1, 1))


def _tool_turn(call):
    return AssistantTurn(text="", tool_calls=(call,), stop_reason="tool_use", usage=Usage(1, 1))


def test_captures_tool_calls_outputs_and_answer():
    call = ToolCall(id="c1", name="run_backtest", input={"model_id": "m1"})
    provider = StubProvider([_tool_turn(call), _final("Sharpe was 1.5.")])

    result = run_candidate("How did m1 do?", provider, Budget(), _TOOLS, _fake_dispatch)

    assert isinstance(result, RunResult)
    assert [c.name for c in result.tool_calls] == ["run_backtest"]
    assert result.tool_outputs == ['{"metrics": {"sharpe": 1.5}}']
    assert result.final_answer == "Sharpe was 1.5."


def test_no_tool_call_run():
    provider = StubProvider([_final("I can only report what the backtests show.")])
    result = run_candidate("Live BTC price?", provider, Budget(), _TOOLS, _fake_dispatch)
    assert result.tool_calls == []
    assert result.tool_outputs == []
    assert "report" in result.final_answer


def test_error_tool_output_not_added_to_pool():
    call = ToolCall(id="c1", name="run_backtest", input={})
    provider = StubProvider([_tool_turn(call), _final("done")])

    def bad_dispatch(name, args):
        return "boom", True

    result = run_candidate("x", provider, Budget(), _TOOLS, bad_dispatch)
    assert result.tool_outputs == []  # errors excluded from the grounding pool
    assert [c.name for c in result.tool_calls] == ["run_backtest"]
