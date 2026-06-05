import json
from pathlib import Path

from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, Usage
from tradinglib.dataset.build import build_dataset
from tradinglib.dataset.scenarios import Scenario


def _final(text):
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(1, 1))


def test_writes_split_jsonl_and_drops_ungrounded(tmp_path: Path):
    scenarios = [
        Scenario("refusal", "Should I buy SPY?", "", None, "2020-01-01", "2020-12-31"),
        Scenario("refusal", "Buy now?", "", None, "2020-01-01", "2020-12-31"),
    ]

    def provider_factory():
        return StubProvider([_final("I can't give buy/sell advice.")])

    stats = build_dataset(
        out_dir=tmp_path,
        scenarios=scenarios,
        provider_factory=provider_factory,
        eval_frac=0.5,
        seed=0,
    )
    train = (tmp_path / "train.jsonl").read_text().splitlines()
    eval_ = (tmp_path / "eval.jsonl").read_text().splitlines()
    assert stats["kept"] == len(train) + len(eval_)
    assert all("messages" in json.loads(line) for line in train + eval_)


def test_ungrounded_trace_is_dropped(tmp_path: Path):
    scenarios = [Scenario("explain", "How did m do?", "", None, "2020-01-01", "2020-12-31")]

    def provider_factory():
        return StubProvider([_final("Sharpe was 2.31.")])  # number, no tool output

    stats = build_dataset(tmp_path, scenarios, provider_factory, eval_frac=0.0, seed=0)
    assert stats["kept"] == 0
    assert stats["dropped_ungrounded"] == 1
