"""Run the eval gate over held-out eval.jsonl and exit 0 (pass) / 1 (fail).

Usage:
    # CI smoke (no GPU, no key): canned provider + stub judge
    python scripts/eval_assistant.py --eval-file data/dataset/eval.jsonl --provider stub

    # real gate in WSL2 after training (needs ANTHROPIC_API_KEY for the judge):
    ANTHROPIC_API_KEY=... python scripts/eval_assistant.py \
        --eval-file data/dataset/eval.jsonl --provider local --adapter out/adapter \
        --out eval_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tradinglib.assistant.types import AssistantTurn, Usage
from tradinglib.assistant_eval.cases import load_eval_cases
from tradinglib.assistant_eval.config import ShipBar
from tradinglib.assistant_eval.judge import StubJudge
from tradinglib.assistant_eval.report import score_run
from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.tools import make_dataset_tools


class _CannedProvider:
    """No-tool, fixed-answer provider so --provider stub runs without GPU or a key."""

    def complete(self, system, conversation, tools) -> AssistantTurn:
        return AssistantTurn(
            text="(stub answer — no model loaded)",
            tool_calls=(),
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the assistant eval gate.")
    p.add_argument("--eval-file", required=True)
    p.add_argument("--provider", choices=["stub", "claude", "local"], default="stub")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (local provider)")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out", default="eval_report.md")
    p.add_argument("--tool-call-bar", type=float, default=0.90)
    p.add_argument("--grounded-bar", type=float, default=0.99)
    p.add_argument("--judge-bar", type=float, default=0.45)
    return p.parse_args(argv)


def build_bar(args: argparse.Namespace) -> ShipBar:
    return ShipBar(
        tool_call_accuracy=args.tool_call_bar,
        grounded_fraction=args.grounded_bar,
        judge_winrate=args.judge_bar,
    )


def _build_provider_and_judge(args):
    if args.provider == "stub":
        return _CannedProvider(), StubJudge(["tie"] * 100_000)
    from tradinglib.assistant_eval.judge import ClaudeJudge

    if args.provider == "claude":
        from tradinglib.assistant.provider import ClaudeProvider

        return ClaudeProvider(), ClaudeJudge()
    from tradinglib.assistant.local_provider import LocalAdapterProvider

    if not args.adapter:
        raise SystemExit("--provider local requires --adapter")
    return LocalAdapterProvider(
        adapter_path=args.adapter, base_model=args.base_model
    ), ClaudeJudge()


def run(args: argparse.Namespace) -> int:
    cases = load_eval_cases(args.eval_file)
    provider, judge = _build_provider_and_judge(args)
    tool_specs, dispatch = make_dataset_tools(Corpus.build())
    bar = build_bar(args)
    card = score_run(cases, provider, judge, tool_specs, dispatch, bar)
    md = card.render()
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    passed = card.passes(bar)
    print(f"\nSHIP-BAR: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
