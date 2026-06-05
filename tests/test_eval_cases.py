import json

from tradinglib.assistant_eval.cases import EvalCase, load_eval_cases


def _row():
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "How did the momentum model do on AAPL?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "run_backtest",
                            "arguments": json.dumps(
                                {"model_id": "m1", "start": "2020-01-01", "end": "2021-01-01"}
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"metrics": {"sharpe": 1.2}}'},
            {"role": "assistant", "content": "Sharpe was 1.2."},
        ],
        "meta": {"category": "explain", "model_id": "m1"},
    }


def test_parses_prompt_gold_calls_and_answer(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    cases = load_eval_cases(p)
    assert len(cases) == 1
    case = cases[0]
    assert isinstance(case, EvalCase)
    assert case.user_prompt == "How did the momentum model do on AAPL?"
    assert case.category == "explain"
    assert case.model_id == "m1"
    assert case.gold_answer == "Sharpe was 1.2."
    assert len(case.gold_tool_calls) == 1
    assert case.gold_tool_calls[0].name == "run_backtest"
    assert case.gold_tool_calls[0].input["model_id"] == "m1"


def test_refusal_case_has_no_tool_calls(tmp_path):
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What's the live BTC price?"},
            {"role": "assistant", "content": "I can only report what the backtests show."},
        ],
        "meta": {"category": "refusal", "model_id": ""},
    }
    p = tmp_path / "eval.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")

    case = load_eval_cases(p)[0]
    assert case.gold_tool_calls == []
    assert case.category == "refusal"


def test_skips_malformed_lines(tmp_path):
    good = json.dumps(_row())
    bad = json.dumps({"messages": [{"role": "user", "content": "x"}]})  # last not assistant
    p = tmp_path / "eval.jsonl"
    p.write_text(good + "\n" + bad + "\n", encoding="utf-8")

    assert len(load_eval_cases(p)) == 1


def test_row_without_user_message_yields_empty_prompt(tmp_path):
    # validate_example permits system+assistant (last is assistant); the loader
    # must tolerate a missing user turn rather than crash.
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hello"},
        ],
        "meta": {"category": "explain", "model_id": "m1"},
    }
    p = tmp_path / "eval.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")

    case = load_eval_cases(p)[0]
    assert case.user_prompt == ""
    assert case.gold_answer == "hello"
