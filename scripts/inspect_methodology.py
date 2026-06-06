"""Compare n12 vs n20 adapter on the methodology cases of dataset_n20/eval.jsonl.

Dumps per-case: gold tool calls, each adapter's calls + score, and a coarse
failure class so we can tell a tool-SELECTION regression from an ARG/format one.
"""

from __future__ import annotations

import json

from tradinglib.assistant.budget import Budget
from tradinglib.assistant.local_provider import LocalAdapterProvider
from tradinglib.assistant_eval.cases import load_eval_cases
from tradinglib.assistant_eval.runner import run_candidate
from tradinglib.assistant_eval.score import _args_match, tool_call_score
from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.tools import make_dataset_tools

EVAL_FILE = "data/dataset_n20/eval.jsonl"
ADAPTERS = {
    "n12": "adapters/qwen25-7b-assistant-n12",
    "n20": "adapters/qwen25-7b-assistant-n20",
}


def _classify(cand, gold) -> str:
    if not gold:
        return "ok(no-gold)" if not cand else "spurious(gold-empty)"
    gold_names = sorted(g.name for g in gold)
    cand_names = sorted(c.name for c in cand)
    if cand_names != gold_names:
        if set(gold_names) == set(cand_names) and len(cand) != len(gold):
            return f"count(gold={len(gold)},cand={len(cand)})"
        return f"name(gold={gold_names},cand={cand_names})"
    # same multiset of names -> check args per matched name
    bad = []
    remaining = list(gold)
    for c in cand:
        for i, g in enumerate(remaining):
            if c.name == g.name:
                if not _args_match(c.input, g.input):
                    bad.append(g.name)
                remaining.pop(i)
                break
    return f"args({','.join(bad)})" if bad else "ok"


def _fmt_calls(calls) -> str:
    return "; ".join(f"{c.name}({json.dumps(c.input, sort_keys=True)})" for c in calls) or "(none)"


def main() -> None:
    cases = [c for c in load_eval_cases(EVAL_FILE) if c.category == "methodology"]
    print(f"methodology cases: {len(cases)}\n")
    tool_specs, dispatch = make_dataset_tools(Corpus.build())

    results = {}  # name -> list of (score, cand_calls, klass)
    for name, path in ADAPTERS.items():
        print(f"### loading {name} ({path})", flush=True)
        provider = LocalAdapterProvider(adapter_path=path, base_model="Qwen/Qwen2.5-7B-Instruct")
        rows = []
        for case in cases:
            run = run_candidate(case.user_prompt, provider, Budget(), tool_specs, dispatch)
            s = tool_call_score(run.tool_calls, case.gold_tool_calls)
            rows.append((s, list(run.tool_calls), _classify(run.tool_calls, case.gold_tool_calls)))
        results[name] = rows
        del provider

    print("\n==== PER-CASE (methodology) ====")
    for idx, case in enumerate(cases):
        s12, c12, k12 = results["n12"][idx]
        s20, c20, k20 = results["n20"][idx]
        flag = (
            "  <-- n20 WORSE"
            if s20 < s12 - 1e-9
            else ("  (n20 better)" if s20 > s12 + 1e-9 else "")
        )
        print(f"\n[{idx}] {case.user_prompt[:90]!r}{flag}")
        print(f"   gold : {_fmt_calls(case.gold_tool_calls)}")
        print(f"   n12  : score={s12:.2f} [{k12}] {_fmt_calls(c12)}")
        print(f"   n20  : score={s20:.2f} [{k20}] {_fmt_calls(c20)}")

    # summary of where n20 lost ground
    worse = [
        (i, results["n20"][i][2])
        for i in range(len(cases))
        if results["n20"][i][0] < results["n12"][i][0] - 1e-9
    ]
    print("\n==== n20 REGRESSIONS (class counts) ====")
    from collections import Counter

    cls = Counter(k.split("(")[0] for _, k in worse)
    print(f"n20 worse on {len(worse)}/{len(cases)} cases:", dict(cls))


if __name__ == "__main__":
    main()
