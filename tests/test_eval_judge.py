import pytest

from tradinglib.assistant_eval.judge import StubJudge, _candidate_is_a, win_rate


def test_candidate_order_is_stable_and_content_derived():
    # same content -> same slot every call (stable across the salted builtin hash)
    assert _candidate_is_a("q", "cand") == _candidate_is_a("q", "cand")
    # the slot depends on content, so at least one differing input flips somewhere
    slots = {_candidate_is_a("q", f"cand{i}") for i in range(8)}
    assert slots == {True, False}


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
