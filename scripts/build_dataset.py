"""Generate the assistant fine-tuning dataset.

Usage:
    ANTHROPIC_API_KEY=... python scripts/build_dataset.py --out data/dataset --n 3
    python scripts/build_dataset.py --out data/dataset --n 1 --limit 5   # smoke

Writes train.jsonl + eval.jsonl. Uses the real Claude teacher; --limit caps the
scenario count for a cheap smoke run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradinglib.assistant.provider import ClaudeProvider
from tradinglib.dataset.build import build_dataset
from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.scenarios import generate_scenarios


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the assistant fine-tuning dataset.")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n", type=int, default=3, help="scenarios per model per category")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-frac", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=None, help="cap total scenarios (smoke)")
    p.add_argument(
        "--split",
        choices=["index", "ticker", "stratified"],
        default="index",
        help="train/eval split: 'index' (shuffle), 'ticker' (strict held-out-by-ticker), "
        "or 'stratified' (by-ticker for explain/counterfactual + random methodology/refusal)",
    )
    return p.parse_args(argv)


def _apply_limit(scenarios, limit):
    return scenarios[:limit] if limit is not None else scenarios


def main() -> None:
    args = parse_args()
    scenarios = _apply_limit(
        generate_scenarios(seed=args.seed, per_model_per_category=args.n), args.limit
    )
    print(f"generating {len(scenarios)} scenarios -> {args.out}")
    stats = build_dataset(
        out_dir=Path(args.out),
        scenarios=scenarios,
        provider_factory=ClaudeProvider,
        eval_frac=args.eval_frac,
        seed=args.seed,
        corpus=Corpus.build(),
        split_by=args.split,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
