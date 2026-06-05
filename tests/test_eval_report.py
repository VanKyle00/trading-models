import json

from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, ToolCall, Usage
from tradinglib.assistant_eval.cases import load_eval_cases
from tradinglib.assistant_eval.config import ShipBar
from tradinglib.assistant_eval.judge import StubJudge
from tradinglib.assistant_eval.report import Scorecard, score_run

_TOOLS = [{"name": "run_backtest", "description": "", "input_schema": {"type": "object"}}]


def _fake_dispatch(name, args):
    return '{"metrics": {"sharpe": 1.5}}', False


def _final(text):
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(1, 1))


def _tool_turn(call):
    return AssistantTurn(text="", tool_calls=(call,), stop_reason="tool_use", usage=Usage(1, 1))


def _write_two_cases(tmp_path):
    row1 = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "How did m1 do?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "run_backtest",
                            "arguments": json.dumps({"model_id": "m1"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"sharpe": 1.5}'},
            {"role": "assistant", "content": "Sharpe was 1.5."},
        ],
        "meta": {"category": "explain", "model_id": "m1"},
    }
    row2 = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Live BTC price?"},
            {"role": "assistant", "content": "I can only report what the backtests show."},
        ],
        "meta": {"category": "refusal", "model_id": ""},
    }
    p = tmp_path / "eval.jsonl"
    p.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
    return p


def test_score_run_aggregates_all_axes(tmp_path):
    cases = load_eval_cases(_write_two_cases(tmp_path))
    provider = StubProvider(
        [
            _tool_turn(ToolCall(id="c1", name="run_backtest", input={"model_id": "m1"})),
            _final("Sharpe was 1.5."),
            _final("I can only report what the backtests show."),
        ]
    )
    judge = StubJudge(["win", "tie"])

    card = score_run(cases, provider, judge, _TOOLS, _fake_dispatch)

    assert isinstance(card, Scorecard)
    assert card.n == 2
    assert card.tool_call_accuracy == 1.0
    assert card.grounded_fraction == 1.0
    assert card.judge_winrate == (1 + 0.5) / 2
    assert set(card.per_category) == {"explain", "refusal"}


def test_passes_checks_every_threshold(tmp_path):
    cases = load_eval_cases(_write_two_cases(tmp_path))
    provider = StubProvider(
        [
            _tool_turn(ToolCall(id="c1", name="run_backtest", input={"model_id": "m1"})),
            _final("Sharpe was 1.5."),
            _final("I can only report what the backtests show."),
        ]
    )
    # win + tie → judge_winrate = 0.75; passes bar of 0.45 but not 0.99
    card = score_run(cases, provider, StubJudge(["win", "tie"]), _TOOLS, _fake_dispatch)
    assert card.passes(ShipBar(0.9, 0.99, 0.45)) is True
    assert card.passes(ShipBar(0.9, 0.99, 0.99)) is False  # judge bar unmet


def test_render_returns_markdown(tmp_path):
    cases = load_eval_cases(_write_two_cases(tmp_path))
    provider = StubProvider(
        [
            _tool_turn(ToolCall(id="c1", name="run_backtest", input={"model_id": "m1"})),
            _final("Sharpe was 1.5."),
            _final("I can only report what the backtests show."),
        ]
    )
    card = score_run(cases, provider, StubJudge(["win", "tie"]), _TOOLS, _fake_dispatch)
    md = card.render()
    assert "tool-call accuracy" in md.lower()
    assert "explain" in md
