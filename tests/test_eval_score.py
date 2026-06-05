from tradinglib.assistant.types import ToolCall
from tradinglib.assistant_eval.runner import RunResult
from tradinglib.assistant_eval.score import grounded, tool_call_score


def _c(name, **inp):
    return ToolCall(id="x", name=name, input=inp)


def test_exact_match_scores_one():
    gold = [_c("run_backtest", model_id="m1", start="2020-01-01")]
    cand = [_c("run_backtest", model_id="m1", start="2020-01-01")]
    assert tool_call_score(cand, gold) == 1.0


def test_numeric_arg_within_tolerance_matches():
    assert (
        tool_call_score([_c("run_backtest", fee_bps=10.0)], [_c("run_backtest", fee_bps=10.1)])
        == 1.0
    )


def test_key_order_independent_and_case_insensitive_strings():
    gold = [_c("run_backtest", model_id="m1", symbol="AAPL")]
    cand = [_c("run_backtest", symbol="aapl", model_id="m1")]
    assert tool_call_score(cand, gold) == 1.0


def test_wrong_args_do_not_match():
    assert (
        tool_call_score([_c("run_backtest", model_id="m2")], [_c("run_backtest", model_id="m1")])
        == 0.0
    )


def test_spurious_call_is_penalized():
    gold = [_c("run_backtest", model_id="m1")]
    cand = [_c("run_backtest", model_id="m1"), _c("list_models")]
    assert tool_call_score(cand, gold) == 0.5  # 1 matched / max(1, 2)


def test_empty_gold_scores_one_only_if_candidate_also_empty():
    assert tool_call_score([], []) == 1.0
    assert tool_call_score([_c("list_models")], []) == 0.0


def test_grounded_delegates_to_verifier():
    run = RunResult(final_answer="Sharpe was 1.5.", tool_outputs=['{"sharpe": 1.5}'])
    assert grounded(run) is True
    bad = RunResult(final_answer="Return was 99.9%.", tool_outputs=['{"sharpe": 1.5}'])
    assert grounded(bad) is False
