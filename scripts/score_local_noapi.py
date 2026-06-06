"""API-free local scoring: tool-call accuracy + grounded fraction per category.

Unlike eval_assistant.py this does NOT run the Claude judge, so it costs no API
credits. Use it to compare adapter checkpoints (e.g. an epoch sweep) on the
deterministic metrics. Pass one or more adapter paths as argv.

    uv run python scripts/score_local_noapi.py data/dataset_n20/eval.jsonl \
        adapters/.../checkpoint-48 adapters/.../checkpoint-72
"""

from __future__ import annotations

import sys
from collections import defaultdict

from tradinglib.assistant.budget import Budget
from tradinglib.assistant.local_provider import LocalAdapterProvider
from tradinglib.assistant_eval.cases import load_eval_cases
from tradinglib.assistant_eval.runner import run_candidate
from tradinglib.assistant_eval.score import grounded, tool_call_score
from tradinglib.dataset.corpus import Corpus
from tradinglib.dataset.tools import make_dataset_tools


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def score(adapter: str, cases, tool_specs, dispatch) -> dict:
    provider = LocalAdapterProvider(adapter_path=adapter, base_model="Qwen/Qwen2.5-7B-Instruct")
    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: {"tc": [], "gr": []})
    tc_all, gr_all = [], []
    for case in cases:
        run = run_candidate(case.user_prompt, provider, Budget(), tool_specs, dispatch)
        s = tool_call_score(run.tool_calls, case.gold_tool_calls)
        g = grounded(run)
        by_cat[case.category]["tc"].append(s)
        by_cat[case.category]["gr"].append(g)
        tc_all.append(s)
        gr_all.append(g)
    del provider
    return {
        "overall": {"tc": _mean(tc_all), "gr": _mean([float(x) for x in gr_all]), "n": len(cases)},
        "by_cat": {
            c: {"tc": _mean(v["tc"]), "gr": _mean([float(x) for x in v["gr"]]), "n": len(v["tc"])}
            for c, v in by_cat.items()
        },
    }


def main() -> None:
    eval_file, adapters = sys.argv[1], sys.argv[2:]
    cases = load_eval_cases(eval_file)
    tool_specs, dispatch = make_dataset_tools(Corpus.build())
    results = {}
    for a in adapters:
        label = a.rstrip("/").split("/")[-1]
        print(f"### scoring {label}", flush=True)
        results[label] = score(a, cases, tool_specs, dispatch)

    cats = sorted({c for r in results.values() for c in r["by_cat"]})
    labels = list(results)
    print("\n==== TOOL-CALL ACCURACY (no judge / no API) ====")
    header = "category".ljust(16) + "".join(lbl.ljust(20) for lbl in labels)
    print(header)
    for cat in cats + ["OVERALL"]:
        row = cat.ljust(16)
        for lbl in labels:
            m = results[lbl]["overall"] if cat == "OVERALL" else results[lbl]["by_cat"].get(cat)
            row += (f"{m['tc']:.3f} (n={m['n']})" if m else "-").ljust(20)
        print(row)
    print("\n==== GROUNDED FRACTION ====")
    for cat in cats + ["OVERALL"]:
        row = cat.ljust(16)
        for lbl in labels:
            m = results[lbl]["overall"] if cat == "OVERALL" else results[lbl]["by_cat"].get(cat)
            row += (f"{m['gr']:.3f}" if m else "-").ljust(20)
        print(row)


if __name__ == "__main__":
    main()
