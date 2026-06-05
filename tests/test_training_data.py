import json
from pathlib import Path

from tradinglib.training.data import (
    load_jsonl,
    to_trl_records,
    validate_dataset,
    validate_example,
)


def _ex(messages):
    return {"messages": messages, "meta": {"category": "explain", "model_id": "m"}}


_GOOD = _ex(
    [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "How did m do?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "run_backtest", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": '{"metrics":{}}'},
        {"role": "assistant", "content": "Sharpe was 0.96."},
    ]
)


def test_load_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(_GOOD) + "\n", encoding="utf-8")
    rows = load_jsonl(p)
    assert len(rows) == 1 and rows[0]["messages"][1]["content"] == "How did m do?"


def test_validate_good_example_has_no_problems():
    assert validate_example(_GOOD) == []


def test_validate_flags_missing_messages():
    assert validate_example({"meta": {}}) != []


def test_validate_flags_bad_role():
    bad = _ex([{"role": "user", "content": "hi"}, {"role": "wizard", "content": "x"}])
    assert any("role" in p for p in validate_example(bad))


def test_validate_flags_tool_without_preceding_tool_call():
    bad = _ex(
        [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "t1", "content": "x"},
        ]
    )
    assert any("tool" in p.lower() for p in validate_example(bad))


def test_validate_dataset_counts(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    bad = _ex([{"role": "user", "content": "hi"}, {"role": "wizard", "content": "x"}])
    p.write_text(json.dumps(_GOOD) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    n_ok, problems = validate_dataset(p)
    assert n_ok == 1 and len(problems) == 1


def test_to_trl_records_strips_meta():
    recs = to_trl_records([_GOOD])
    assert recs == [{"messages": _GOOD["messages"]}]


def test_prepare_records_drops_invalid_and_strips_meta(tmp_path: Path):
    from tradinglib.training.data import prepare_records

    bad = _ex([{"role": "user", "content": "hi"}, {"role": "wizard", "content": "x"}])
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_GOOD) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    records, n_dropped = prepare_records(p)
    assert n_dropped == 1
    assert records == [{"messages": _GOOD["messages"]}]  # meta stripped, bad dropped


def test_prepare_records_raises_on_empty_when_required(tmp_path: Path):
    from tradinglib.training.data import prepare_records

    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        prepare_records(p, require_nonempty=True)


def test_prepare_records_allows_empty_when_not_required(tmp_path: Path):
    from tradinglib.training.data import prepare_records

    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    records, n_dropped = prepare_records(p, require_nonempty=False)
    assert records == [] and n_dropped == 0
