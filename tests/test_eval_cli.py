import json

from scripts.eval_assistant import build_bar, parse_args, run


def test_parse_args_defaults():
    args = parse_args(["--eval-file", "e.jsonl"])
    assert args.provider == "stub"
    assert args.eval_file == "e.jsonl"


def test_build_bar_overrides():
    args = parse_args(["--eval-file", "e.jsonl", "--grounded-bar", "1.0"])
    bar = build_bar(args)
    assert bar.grounded_fraction == 1.0
    assert bar.tool_call_accuracy == 0.90


def test_stub_run_end_to_end_writes_report(tmp_path):
    row = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "Live BTC price?"},
            {"role": "assistant", "content": "I can only report what the backtests show."},
        ],
        "meta": {"category": "refusal", "model_id": ""},
    }
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "report.md"

    code = run(
        parse_args(
            [
                "--eval-file",
                str(eval_file),
                "--provider",
                "stub",
                "--out",
                str(out),
            ]
        )
    )

    assert out.exists()
    assert "Eval scorecard" in out.read_text(encoding="utf-8")
    assert code in (0, 1)  # deterministic given the canned provider; just must not raise
