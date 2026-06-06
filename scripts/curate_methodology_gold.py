"""Curate methodology gold: fix the train/test-split answers in a dataset.

Ground truth (docs/methodology.md): the chronological 80/20 train/test split is
an ML-model discipline. Only models/ml/* (the GBM/XGBoost model) is ML; the
classical / options / alt-data / microstructure models are rule-based with no
fitted parameters, so they have NO train/test split -- they're backtested over
the full period. Both dataset_n12 and dataset_n20 get this wrong for some non-ML
models. This rewrites the gold final answer for every "train/test split"
methodology row to a ground-truth-correct canonical answer (tool calls, tool
outputs, and all other rows are left untouched).

    uv run python scripts/curate_methodology_gold.py data/dataset_n20 data/dataset_n21
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# model display name (as it appears in the question) -> (is_ml, descriptor)
_MODELS = {
    "XGBoost Next-Day Return on SPY": (True, "gradient-boosted ML"),
    "Delta-Hedged Long Option on SPY": (False, "rules-based options"),
    "Google Trends Contrarian on BTC": (
        False,
        "rule-based alternative-data (contrarian sentiment)",
    ),
    "Order Flow Imbalance on BTC": (False, "rule-based market-microstructure"),
    "SMA Crossover on SPY": (False, "classical rule-based (moving-average crossover)"),
}


def _canonical(model_name: str, is_ml: bool, descriptor: str) -> str:
    if is_ml:
        return (
            f"Based on the repo's methodology, the **{model_name}** model uses the default "
            f"**chronological 80/20 train/test split**: the {descriptor} model is fit on the "
            f"earlier 80% of the data and evaluated on the held-out later 20%. The split is "
            f"strictly chronological -- data is never shuffled, so future bars never enter the "
            f"training window -- and the reported metrics are the out-of-sample results from "
            f"that held-out test set."
        )
    return (
        f"Based on the repo's methodology, the **{model_name}** model does not use a train/test "
        f"split. The chronological 80/20 split is an **ML-model** discipline -- it applies to "
        f"models that fit parameters on a training window and are evaluated out-of-sample. This "
        f"is a {descriptor} strategy with no fitted parameters, so there is nothing to hold out: "
        f"it is backtested over its full available period, and the reported metrics are "
        f"full-sample backtest results. The engine still enforces the repo-wide guards -- data is "
        f"strictly chronological (never shuffled) and every signal is lagged one bar to prevent "
        f"look-ahead bias."
    )


def _match_model(question: str) -> tuple[str, bool, str] | None:
    for name, (is_ml, desc) in _MODELS.items():
        if name in question:
            return name, is_ml, desc
    return None


def curate_file(src: Path, dst: Path) -> tuple[int, int]:
    changed = total_meth = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("meta", {}).get("category") == "methodology":
                total_meth += 1
                msgs = row["messages"]
                q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                if "train/test split" in q.lower() and (hit := _match_model(q)):
                    msgs[-1]["content"] = _canonical(*hit)
                    changed += 1
            fout.write(json.dumps(row) + "\n")
    return changed, total_meth


def main() -> None:
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("train.jsonl", "eval.jsonl"):
        c, t = curate_file(src_dir / fname, dst_dir / fname)
        print(f"{fname}: rewrote {c} train/test-split answers ({t} methodology rows total)")
    # carry over dropped.jsonl unchanged if present (not used in training)
    drp = src_dir / "dropped.jsonl"
    if drp.exists():
        shutil.copy(drp, dst_dir / "dropped.jsonl")
    print(f"wrote curated dataset -> {dst_dir}")


if __name__ == "__main__":
    main()
