import sys


def test_module_imports_without_torch():
    # importing the provider module must not pull torch into the process
    import tradinglib.assistant.local_provider  # noqa: F401

    assert "torch" not in sys.modules, "local_provider must keep torch imports lazy"


def test_parses_single_qwen_tool_call():
    from tradinglib.assistant.local_provider import parse_qwen_tool_calls

    text = 'Let me check.\n<tool_call>\n{"name": "run_backtest", "arguments": {"model_id": "m1"}}\n</tool_call>'
    calls, stop_reason, prose = parse_qwen_tool_calls(text)
    assert stop_reason == "tool_use"
    assert len(calls) == 1
    assert calls[0].name == "run_backtest"
    assert calls[0].input == {"model_id": "m1"}
    assert calls[0].id  # a non-empty synthesized id
    assert prose.strip() == "Let me check."


def test_parses_multiple_tool_calls():
    from tradinglib.assistant.local_provider import parse_qwen_tool_calls

    text = (
        '<tool_call>{"name": "list_models", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "get_model_spec", "arguments": {"model_id": "m1"}}</tool_call>'
    )
    calls, stop_reason, _ = parse_qwen_tool_calls(text)
    assert [c.name for c in calls] == ["list_models", "get_model_spec"]
    assert stop_reason == "tool_use"


def test_no_tool_call_is_final_answer():
    from tradinglib.assistant.local_provider import parse_qwen_tool_calls

    calls, stop_reason, prose = parse_qwen_tool_calls("Sharpe was 1.5.")
    assert calls == []
    assert stop_reason == "end_turn"
    assert prose == "Sharpe was 1.5."


def test_malformed_tool_call_json_is_skipped():
    from tradinglib.assistant.local_provider import parse_qwen_tool_calls

    text = "<tool_call>{not valid json}</tool_call>"
    calls, stop_reason, _ = parse_qwen_tool_calls(text)
    assert calls == []
    assert stop_reason == "end_turn"
