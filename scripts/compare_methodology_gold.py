"""Compare the methodology GOLD answers in dataset_n12 vs dataset_n20 (train).

The n20-trained adapter answers methodology questions worse (judge ~0.31) than
the n12-trained one (~0.59) against the SAME eval gold, with epoch controlled.
The remaining difference is the training gold itself -- this dumps it side by
side so we can see the quality/style regression.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict


def load_methodology(path: str) -> dict[str, list[str]]:
    """question -> list of gold final answers, methodology rows only."""
    out: dict[str, list[str]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("meta", {}).get("category") != "methodology":
                continue
            msgs = r["messages"]
            q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            ans = msgs[-1].get("content", "") if msgs else ""
            out[q].append(ans)
    return out


def wc(s: str) -> int:
    return len(s.split())


def summarize(name: str, data: dict[str, list[str]]) -> None:
    answers = [a for v in data.values() for a in v]
    lens = [wc(a) for a in answers]
    print(f"\n## {name}: {len(answers)} methodology gold answers, {len(data)} unique questions")
    if lens:
        print(
            f"   words/answer: mean={statistics.mean(lens):.0f} "
            f"median={statistics.median(lens):.0f} min={min(lens)} max={max(lens)}"
        )


def main() -> None:
    n12 = load_methodology("data/dataset_n12/train.jsonl")
    n20 = load_methodology("data/dataset_n20/train.jsonl")
    summarize("n12 train", n12)
    summarize("n20 train", n20)

    shared = sorted(set(n12) & set(n20))
    print(
        f"\n## shared questions: {len(shared)}  (n12-only={len(set(n12) - set(n20))}, n20-only={len(set(n20) - set(n12))})"
    )
    for i, q in enumerate(shared):
        a12, a20 = n12[q][0], n20[q][0]
        print(f"\n===[{i}] Q: {q}")
        print(f"--- n12 gold ({wc(a12)}w):\n{a12}")
        print(f"--- n20 gold ({wc(a20)}w):\n{a20}")


if __name__ == "__main__":
    main()
