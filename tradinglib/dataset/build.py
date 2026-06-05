"""Orchestrate: scenarios -> record (real tools) -> ground-filter -> serialize ->
train/eval split -> write JSONL. Split is by scenario index after a seeded
shuffle; callers wanting strict held-out tickers/templates pre-partition the
scenario list. The grounding filter drops any trace whose answer cites a number
absent from its tool outputs."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tradinglib.assistant.budget import Budget
from tradinglib.assistant.provider import LLMProvider
from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.grounding import is_grounded
from tradinglib.dataset.recorder import record_trace
from tradinglib.dataset.scenarios import Scenario
from tradinglib.dataset.serialize import to_chat_example
from tradinglib.dataset.tools import make_dataset_tools


def build_dataset(
    out_dir: Path,
    scenarios: list[Scenario],
    provider_factory: Callable[[], LLMProvider],
    eval_frac: float = 0.15,
    seed: int = 0,
    corpus: Corpus | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = corpus or Corpus.from_chunks(["(empty corpus)"])
    tool_specs, dispatch = make_dataset_tools(corpus)

    kept: list[dict[str, Any]] = []
    dropped_failed = dropped_ungrounded = 0

    for scn in scenarios:
        trace = record_trace(scn, provider_factory(), Budget(), tool_specs, dispatch)
        if not trace.ok:
            dropped_failed += 1
            continue
        ok, _missing = is_grounded(trace.final_answer, trace.tool_outputs)
        if not ok:
            dropped_ungrounded += 1
            continue
        kept.append(to_chat_example(trace))

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_eval = int(len(kept) * eval_frac)
    eval_rows, train_rows = kept[:n_eval], kept[n_eval:]

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)

    return {
        "kept": len(kept),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "dropped_failed": dropped_failed,
        "dropped_ungrounded": dropped_ungrounded,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
