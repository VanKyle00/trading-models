import pytest

from tradinglib.assistant_eval.judge import StubJudge, win_rate


def test_win_rate_counts_ties_as_half():
    assert win_rate(["win", "win", "loss", "tie"]) == pytest.approx((2 + 0.5) / 4)


def test_win_rate_empty_is_zero():
    assert win_rate([]) == 0.0


def test_stub_judge_returns_scripted_verdicts():
    judge = StubJudge(["win", "loss"])
    assert judge.compare("q", "gold", "cand") == "win"
    assert judge.compare("q", "gold", "cand") == "loss"


def test_parse_verdict_maps_letters():
    from tradinglib.assistant_eval.judge import _parse_verdict

    # candidate presented as "A": "A" -> win, "B" -> loss, tie -> tie
    assert _parse_verdict("Answer: A", candidate_label="A") == "win"
    assert _parse_verdict("Answer: B", candidate_label="A") == "loss"
    assert _parse_verdict("Answer: B", candidate_label="B") == "win"
    assert _parse_verdict("verdict: TIE", candidate_label="A") == "tie"
    assert _parse_verdict("garbage", candidate_label="A") == "tie"  # unparseable -> tie
