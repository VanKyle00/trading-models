# Training kickoff — point Claude here to run Step A

**Purpose:** orientation + runbook for *running the assistant training* (Step A of the
own-domain-LLM program) once WSL2 is installed. Open a Claude Code session **inside the
WSL2 Ubuntu shell**, from the repo root, and say: *"Read docs/training-kickoff.md and drive Step A."*

---

## Where we are (as of 2026-06-05)

The own-domain LLM assistant is built in four sub-projects: **data → train → eval → serve.**

| # | Sub-project | Status |
|---|-------------|--------|
| 1 | Data harness (`tradinglib/dataset/`) | ✅ on `main` |
| 2 | QLoRA training (`tradinglib/training/`, `scripts/train_assistant.py`) | ✅ on `main` |
| 3 | Eval harness (`tradinglib/assistant_eval/`, `scripts/eval_assistant.py`) | ✅ on `main` (PR #27) |
| 4 | Serving (Modal vLLM + `VLLMProvider`) | ⬜️ after a model clears the gate |

**Step A = run the chain end-to-end:** generate dataset → smoke-train → full QLoRA on the
RTX 5080 → run the eval gate. Everything needed is already on `main`. Detailed reference:
`docs/training-assistant.md` (install, smoke, full run, fallback, handoff) and the eval-gate
section at its end.

The dataset is **self-generated**, not downloaded: `build_dataset.py` enumerates scenarios,
runs the REAL tools (real backtests on real market data), and uses Claude as the teacher to
write prose around the real numbers. Needs `ANTHROPIC_API_KEY`.

---

## Prerequisites to confirm FIRST (inside WSL2)

Claude: verify these before doing anything expensive. Stop and report if any fail.

```bash
nvidia-smi                      # must list the RTX 5080 INSIDE wsl (GPU passthrough working)
nvcc --version                  # CUDA 12.8+ toolkit visible in wsl (Blackwell sm_120 needs it)
echo "${ANTHROPIC_API_KEY:0:7}" # must be non-empty — required for dataset generation
uv --version                    # uv present
```

If `nvidia-smi` works on the Windows host but not in WSL2, the NVIDIA driver/passthrough
isn't wired yet — fix that before continuing (it's a host-side driver matter, not a repo issue).

---

## The run sequence (Claude: drive this, verifying each gate)

Work from the repo root on `main` (pull latest first: `git pull`). Prefer running from a clone
inside the WSL filesystem (e.g. `~/trading-models`) rather than `/mnt/c/...` for much faster I/O;
if you clone fresh, this file is on `main` so it'll be there.

```bash
# 0. Install training deps (once). Then Unsloth per github.com/unslothai/unsloth (latest, for sm_120).
uv pip install -e ".[train]"

# 1. Generate the dataset (real Claude teacher + real backtests). Smoke it small FIRST:
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uv run python scripts/build_dataset.py --out data/dataset --n 3 --limit 8
#    → inspect data/dataset/{train,eval}.jsonl look sane, then full:
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uv run python scripts/build_dataset.py --out data/dataset --n 3

# 2. Smoke the trainer (10 steps) — proves 4-bit load → LoRA attach → SFTTrainer → save:
uv run python scripts/train_assistant.py --train data/dataset/train.jsonl --eval data/dataset/eval.jsonl --out adapters/smoke --max-steps 10

# 3. Full QLoRA run (2 epochs):
uv run python scripts/train_assistant.py --train data/dataset/train.jsonl --eval data/dataset/eval.jsonl --out adapters/qwen25-7b-assistant

# 4. The eval gate (needs the key for the Claude judge):
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY uv run python scripts/eval_assistant.py --eval-file data/dataset/eval.jsonl --provider local --adapter adapters/qwen25-7b-assistant --out eval_report.md
```

Between steps: confirm the expected artifact exists before moving on (`data/dataset/*.jsonl`;
`adapters/smoke/adapter_model.safetensors`; `adapters/qwen25-7b-assistant/...`; `eval_report.md`).

---

## Known risks to watch (don't re-derive these live)

1. **TRL API drift** (step 2/3). If `SFTTrainer` errors on a kwarg: newer TRL wants
   `processing_class=tokenizer` not `tokenizer=tokenizer`; and `SFTConfig` field
   `eval_strategy` vs `evaluation_strategy` varies by version. Check installed versions, adjust
   `scripts/train_assistant.py`. Fallback (TRL+peft+bnb, if Unsloth has no sm_120 wheel) is in
   `docs/training-assistant.md`.
2. **Qwen tool-call parser** (step 4). `tradinglib/assistant/local_provider.py:parse_qwen_tool_calls`
   is the one piece CI couldn't cover — it parses the model's actual `<tool_call>{json}</tool_call>`
   output. If `eval_report.md`'s tool-call column is ~0 across `explain`/`counterfactual` cases,
   the model is likely emitting a different format; inspect a raw generation and adjust the parser.
3. **VRAM** (step 3). Defaults (seq 2048, batch 2, grad-accum 4) target 16 GB. On OOM: add
   `--max-seq-len 1024`, then reduce batch in `tradinglib/training/config.py`.
4. **Eval-split leakage caveat.** The train/eval split is currently index-based after a shuffle,
   NOT strict held-out-by-ticker — so a ticker/template pattern can leak train→eval, inflating
   the gate's generalization read. Ask the user whether to apply the by-ticker pre-partition fix
   before trusting the numbers (it's a small change in `dataset/build.py` + `scenarios.py`).

---

## The decision at the end (the ship-bar)

`eval_assistant.py` exits `0` if the candidate clears: **tool-call accuracy ≥0.90, grounded
≥0.99, judge win-rate ≥0.45** (and exits `1` otherwise). Read `eval_report.md` (overall +
per-category) WITH the user and decide:

- **Clears the bar** → ship as default; proceed to **sub-project 4** (serve on Modal vLLM via a
  new `VLLMProvider`, add runtime `search_docs`, router-ready).
- **Close but under** → ship with the router keeping Claude on hard queries.
- **Far off** → diagnose the weak axis (tool-call → consider a normalized public function-calling
  supplement; methodology/grounding → expand the RAG corpus; narrow/overfit → widen real tickers)
  and/or take the documented Qwen2.5-14B upgrade path.

Design rationale and the full plan: `docs/superpowers/specs/2026-06-04-own-domain-llm-assistant-design.md`
and the per-sub-project plans under `docs/superpowers/plans/` (local-only, gitignored).
