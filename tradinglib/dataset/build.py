"""Orchestrate: scenarios -> record (real tools) -> ground-filter -> serialize ->
train/eval split -> write JSONL. Split is either by scenario index after a seeded
shuffle (``split_by="index"``, the default) or strict held-out-by-ticker
(``split_by="ticker"``) so no ticker leaks train->eval. The grounding filter
drops any trace whose answer cites a number absent from its tool outputs; every
drop (failed or ungrounded) is logged to ``dropped.jsonl`` for diagnosis."""

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
from tradinglib.dataset.scenarios import Scenario, partition_by_ticker, partition_stratified
from tradinglib.dataset.serialize import to_chat_example
from tradinglib.dataset.tools import make_dataset_tools

_MAX_LOG_FIELD = 1200  # cap logged answer/tool-output strings so dropped.jsonl stays readable
_ABORT_AFTER_CONSEC_FAILS = 8  # a burst of failures = API outage (credits/rate/key), not bad luck


def _record_and_filter(
    scenarios: list[Scenario],
    provider_factory: Callable[[], LLMProvider],
    tool_specs: list[dict],
    dispatch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Record + ground-filter a scenario group. Returns (kept_rows, dropped_log,
    n_failed, n_ungrounded). ``dropped_log`` carries enough context (scenario,
    reason, missing numbers, answer, tool outputs) to diagnose why drops happen."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    n_failed = n_ungrounded = 0
    consec_fail = 0

    for scn in scenarios:
        trace = record_trace(scn, provider_factory(), Budget(), tool_specs, dispatch)
        scn_meta = {
            "category": scn.category,
            "model_id": scn.model_id,
            "symbol": scn.symbol,
            "question": scn.question,
        }
        if not trace.ok:
            n_failed += 1
            consec_fail += 1
            # A run of failures means the provider is down (out of credits, rate
            # limited, bad key) -- abort loudly rather than silently writing a
            # broken, partial dataset (the failure path otherwise looks like a
            # normal "model didn't answer" drop).
            if consec_fail >= _ABORT_AFTER_CONSEC_FAILS:
                raise RuntimeError(
                    f"Aborting dataset build: {consec_fail} consecutive trace failures "
                    f"after {n_failed + len(kept) + n_ungrounded} scenarios. This is almost "
                    f"certainly an API problem (credits, rate limit, or key), not the data. "
                    f"Last provider error: {trace.error or 'unknown'}"
                )
            dropped.append({"reason": "failed", "scenario": scn_meta, "error": trace.error})
            continue
        consec_fail = 0
        ok, missing = is_grounded(trace.final_answer, trace.tool_outputs)
        if not ok:
            n_ungrounded += 1
            dropped.append(
                {
                    "reason": "ungrounded",
                    "scenario": scn_meta,
                    "missing": missing,
                    "answer": trace.final_answer[:_MAX_LOG_FIELD],
                    "tool_outputs": [o[:_MAX_LOG_FIELD] for o in trace.tool_outputs],
                }
            )
            continue
        kept.append(to_chat_example(trace))

    return kept, dropped, n_failed, n_ungrounded


def build_dataset(
    out_dir: Path,
    scenarios: list[Scenario],
    provider_factory: Callable[[], LLMProvider],
    eval_frac: float = 0.15,
    seed: int = 0,
    corpus: Corpus | None = None,
    split_by: str = "index",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = corpus or Corpus.from_chunks(["(empty corpus)"])
    tool_specs, dispatch = make_dataset_tools(corpus)

    if split_by in ("ticker", "stratified"):
        if split_by == "ticker":
            train_scn, eval_scn = partition_by_ticker(scenarios, eval_frac, seed)
        else:
            train_scn, eval_scn = partition_stratified(scenarios, eval_frac, seed)
        train_rows, drop_t, fail_t, ung_t = _record_and_filter(
            train_scn, provider_factory, tool_specs, dispatch
        )
        eval_rows, drop_e, fail_e, ung_e = _record_and_filter(
            eval_scn, provider_factory, tool_specs, dispatch
        )
        dropped_log = drop_t + drop_e
        dropped_failed, dropped_ungrounded = fail_t + fail_e, ung_t + ung_e
    elif split_by == "index":
        kept, dropped_log, dropped_failed, dropped_ungrounded = _record_and_filter(
            scenarios, provider_factory, tool_specs, dispatch
        )
        rng = random.Random(seed)
        rng.shuffle(kept)
        n_eval = int(len(kept) * eval_frac)
        eval_rows, train_rows = kept[:n_eval], kept[n_eval:]
    else:
        raise ValueError(
            f"unknown split_by: {split_by!r} (expected 'index', 'ticker', or 'stratified')"
        )

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "dropped.jsonl", dropped_log)

    return {
        "kept": len(train_rows) + len(eval_rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "dropped_failed": dropped_failed,
        "dropped_ungrounded": dropped_ungrounded,
        "split_by": split_by,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
