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


def test_extra_optional_args_on_candidate_still_match():
    # The model often spells out default/inert knobs the minimal gold omits;
    # that over-specification is not a wrong tool call.
    gold = [_c("run_backtest", model_id="m1", symbol="IWM", start="2020-01-01")]
    cand = [
        _c(
            "run_backtest",
            model_id="m1",
            symbol="IWM",
            start="2020-01-01",
            fee_bps=1,
            slippage_bps=5,
            params={"implied_vol": 0.18},
        )
    ]
    assert tool_call_score(cand, gold) == 1.0


def test_missing_gold_arg_still_fails():
    # but if gold specified a knob (e.g. fees for a "fees are 10 bps" question),
    # the candidate must include it
    gold = [_c("run_backtest", model_id="m1", fee_bps=10.0)]
    cand = [_c("run_backtest", model_id="m1")]
    assert tool_call_score(cand, gold) == 0.0


def test_spurious_call_is_penalized():
    gold = [_c("run_backtest", model_id="m1")]
    cand = [_c("run_backtest", model_id="m1"), _c("list_models")]
    assert tool_call_score(cand, gold) == 0.5  # 1 matched / max(1, 2)


def test_empty_gold_scores_one_only_if_candidate_also_empty():
    assert tool_call_score([], []) == 1.0
    assert tool_call_score([_c("list_models")], []) == 0.0


def test_query_paraphrase_matches():
    # Same tool, semantically-equivalent query wording: one substituted content
    # word ("fee" vs "execution") must not score the call as a miss.
    gold = [_c("search_docs", query="trade fills slippage execution model")]
    cand = [_c("search_docs", query="trade fills slippage fee model")]
    assert tool_call_score(cand, gold) == 1.0


def test_query_overspecific_gold_matches():
    # Gold query carries extra descriptor words the model need not reproduce.
    gold = [_c("search_docs", query="Deflated Sharpe ratio definition meaning")]
    cand = [_c("search_docs", query="Deflated Sharpe ratio")]
    assert tool_call_score(cand, gold) == 1.0


def test_query_exact_still_matches():
    gold = [_c("search_docs", query="Order Flow Imbalance BTC data source")]
    cand = [_c("search_docs", query="Order Flow Imbalance BTC data source")]
    assert tool_call_score(cand, gold) == 1.0


def test_query_different_topic_does_not_match():
    # Different model entirely: low overlap must still fail.
    gold = [_c("search_docs", query="SMA Crossover SPY data source")]
    cand = [_c("search_docs", query="Order Flow Imbalance BTC data source")]
    assert tool_call_score(cand, gold) == 0.0


def test_query_different_question_same_model_does_not_match():
    # Same model, different question (data source vs train/test split): the
    # overlap stays below threshold so these stay distinct.
    gold = [_c("search_docs", query="Order Flow Imbalance BTC data source")]
    cand = [_c("search_docs", query="Order Flow Imbalance BTC train test split")]
    assert tool_call_score(cand, gold) == 0.0


def test_non_query_string_arg_stays_exact():
    # The overlap loosening is scoped to free-text keys; identifier args still
    # require an exact match.
    gold = [_c("get_model_spec", model_id="models/classical/01-sma-crossover-spy")]
    cand = [_c("get_model_spec", model_id="models/classical/02-sma-crossover-spy")]
    assert tool_call_score(cand, gold) == 0.0


def test_grounded_delegates_to_verifier():
    run = RunResult(final_answer="Sharpe was 1.5.", tool_outputs=['{"sharpe": 1.5}'])
    assert grounded(run) is True
    bad = RunResult(final_answer="Return was 99.9%.", tool_outputs=['{"sharpe": 1.5}'])
    assert grounded(bad) is False
