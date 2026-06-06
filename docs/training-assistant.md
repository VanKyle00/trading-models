# QLoRA fine-tuning the assistant model (sub-project 2)

Produces a LoRA adapter from the harness dataset. Serving the adapter (Modal + vLLM) is sub-project 4.

---

## Why WSL2

The RTX 5080 is Blackwell (sm_120). The training stack — bitsandbytes, Unsloth, Triton — requires CUDA 12.8+ and Linux kernel GPU passthrough. These packages either ship broken wheels or need source builds on native Windows. Run everything in this doc inside a WSL2 Ubuntu session.

---

## Prerequisites

| Requirement | Check |
| --- | --- |
| NVIDIA driver on Windows (≥ 560 recommended for Blackwell + WSL2 passthrough) | `nvidia-smi` on the Windows host |
| CUDA 12.8+ toolkit visible inside WSL2 | `nvcc --version` in WSL2 |
| RTX 5080 listed in WSL2 | `nvidia-smi` in WSL2 — should show the 5080 |

WSL2 GPU passthrough is handled by the NVIDIA driver on the Windows side; no separate CUDA install is needed inside WSL2 beyond the toolkit (the driver-side runtime is projected into WSL2 automatically).

---

## Install (WSL2 only)

From the repo root inside WSL2:

```bash
# 1. Install training extras — pulls in transformers, trl, peft, datasets,
#    accelerate, bitsandbytes, torch (CUDA build).
uv pip install -e ".[train]"

# 2. Install Unsloth — the build must match the installed torch + CUDA version.
#    Blackwell/sm_120 support is recent: always use the latest release.
#    Follow the current official instructions at:
#    https://github.com/unslothai/unsloth
```

**Do not install `[train]` in the base environment or the Modal deploy environment.** The heavy GPU deps (bitsandbytes, Triton, Unsloth) are local-only.

---

## Generate the dataset (sub-project 1)

Sub-project 1 produces `train.jsonl` + `eval.jsonl` in `data/dataset/`. Run it before training.

```bash
# Full generation (real Claude teacher; uses your Anthropic key):
ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/build_dataset.py \
    --out data/dataset --n 3

# Cheap first pass — cap total scenarios to 8:
ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/build_dataset.py \
    --out data/dataset --n 3 --limit 8
```

**Known limitation — eval split:** the current split is index-based after a shuffle (15 % held out by `--eval-frac`). For a rigorous held-out eval, pre-partition scenarios by ticker/template *before* any shuffle so no ticker/template pattern leaks from train to eval. Address this before a production-quality run.

---

## Smoke run (10 steps)

Proves the whole pipeline: 4-bit load → LoRA attach → SFTTrainer → adapter save.

```bash
uv run python scripts/train_assistant.py \
    --train data/dataset/train.jsonl \
    --eval  data/dataset/eval.jsonl \
    --out   adapters/smoke \
    --max-steps 10
```

Expected output:
- Unsloth loads `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` in 4-bit.
- LoRA is attached (r=16, alpha=32, 7 target modules).
- 10 training steps run; eval loss is reported if `--eval` is provided.
- `adapters/smoke/` contains `adapter_model.safetensors` and `adapter_config.json`.

### API-drift check (important)

Unsloth and TRL move fast. If the script errors on a call signature, verify against your installed versions and adjust `scripts/train_assistant.py`. Two known drift points to check first:

- **`SFTTrainer` tokenizer kwarg** — newer TRL may want `processing_class=tokenizer` instead of `tokenizer=tokenizer`.
- **`SFTConfig` field names** — `eval_strategy` (used in the script) vs `evaluation_strategy`; field names vary by TRL version.

---

## Full run

Drop `--max-steps` to run both epochs (default: 2 epochs, lr 2e-4):

```bash
uv run python scripts/train_assistant.py \
    --train data/dataset/train.jsonl \
    --eval  data/dataset/eval.jsonl \
    --out   adapters/qwen25-7b-assistant
```

**Runtime:** expect minutes to low hours on the 5080.

## Checkpoint selection (epoch sweep)

`load_best_model_at_end` keeps the lowest-`eval_loss` checkpoint. That's a decent
default, but **eval_loss under-selects for tool-call accuracy**: on our data the
eval_loss minimum lands at ~epoch 2, while tool-call formatting only fully locks
in at ~epoch 3 (where eval_loss is barely higher). In one measured run methodology
tool-call went 0.781 → 0.969 between epoch 2 and 3, and overall 0.602 → 0.675, for
a negligible eval_loss change (0.581 → 0.586).

For a model that ships on the eval gate, sweep the epochs instead of trusting
eval_loss:

```bash
# 1. keep every epoch checkpoint
uv run python scripts/train_assistant.py \
    --train data/dataset/train.jsonl --eval data/dataset/eval.jsonl \
    --out adapters/run-allck --epochs 8 --save-total-limit 0

# 2. score candidate epochs WITHOUT the Claude judge (no API spend): tool-call + grounded
uv run python scripts/score_local_noapi.py data/dataset/eval.jsonl \
    adapters/run-allck/checkpoint-48 adapters/run-allck/checkpoint-72 adapters/run-allck/checkpoint-96

# 3. promote the best epoch's inference files to a clean adapter dir and point
#    ASSISTANT_ADAPTER (webapp) / --adapter (eval) at it.
```

Pick the epoch with the best tool-call near the eval_loss minimum; confirm answer
quality with the full eval gate (`eval_assistant.py`, needs the judge / API) before shipping.

**VRAM:** the defaults (seq 2048, batch 2, grad-accum 4) target 16 GB. If you hit OOM:

```bash
# Lower sequence length:
uv run python scripts/train_assistant.py \
    --train data/dataset/train.jsonl \
    --eval  data/dataset/eval.jsonl \
    --out   adapters/qwen25-7b-assistant \
    --max-seq-len 1024
```

Further OOM: reduce `per_device_train_batch_size` by editing `TrainSettings` in `tradinglib/training/config.py` (or add a `--batch-size` flag to the script).

**Output:** `adapters/qwen25-7b-assistant/` — `adapter_model.safetensors` + `adapter_config.json`. This is the artifact that sub-project 4 consumes.

---

## Fallback: TRL + peft + bitsandbytes (if Unsloth fails on Blackwell)

If Unsloth has no wheel for your sm_120 build, swap the model-load block in `scripts/train_assistant.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",      # HF hub id (not the unsloth variant)
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

lora_cfg = LoraConfig(
    r=settings.lora.r,
    lora_alpha=settings.lora.alpha,
    lora_dropout=settings.lora.dropout,
    target_modules=list(settings.lora.target_modules),
    bias=settings.lora.bias,
)
model = get_peft_model(model, lora_cfg)
```

Everything after this point — `SFTConfig`, `SFTTrainer`, `trainer.train()`, `model.save_pretrained()` — is identical to the Unsloth path.

---

## Handoff to sub-project 4

This sub-project ends with a local adapter directory and a reported eval loss.

Sub-project 4 picks up the adapter and:
1. Pushes `adapters/qwen25-7b-assistant/` to a Modal Volume or HF Hub.
2. Serves base + adapter via vLLM (`--lora-modules`) on Modal.
3. Wires a new `LLMProvider` implementation that points at the vLLM endpoint.

No changes to `tradinglib/training/` are needed for sub-project 4.

---

## Quick reference

| Command | What it does |
| --- | --- |
| `uv pip install -e ".[train]"` | Install training deps (WSL2 only) |
| `ANTHROPIC_API_KEY=... uv run python scripts/build_dataset.py --out data/dataset --n 3` | Generate full dataset (sub-project 1) |
| `ANTHROPIC_API_KEY=... uv run python scripts/build_dataset.py --out data/dataset --n 3 --limit 8` | Quick dataset smoke run (8 scenarios) |
| `uv run python scripts/train_assistant.py --train data/dataset/train.jsonl --eval data/dataset/eval.jsonl --out adapters/smoke --max-steps 10` | 10-step smoke run |
| `uv run python scripts/train_assistant.py --train data/dataset/train.jsonl --eval data/dataset/eval.jsonl --out adapters/qwen25-7b-assistant` | Full training run (2 epochs) |

---

## Running the eval gate (sub-project 3)

After a training run produces an adapter, run the gate in WSL2 against the held-out
`eval.jsonl`. The judge calls Claude, so export your key:

```bash
ANTHROPIC_API_KEY=... uv run python scripts/eval_assistant.py \
    --eval-file data/dataset/eval.jsonl \
    --provider local --adapter adapters/qwen25-7b-assistant \
    --out eval_report.md
```

Exit code `0` means the candidate cleared the ship-bar (tool-call accuracy ≥0.90,
grounded ≥0.99, judge win-rate ≥0.45) and can become the default provider; `1` means
keep Claude as default (or enable the router later). The scorecard (overall + per-category)
is printed and written to `--out`.

CI smoke (no GPU, no key) — exercises the whole pipeline with a canned provider:

```bash
python scripts/eval_assistant.py --eval-file data/dataset/eval.jsonl --provider stub
```

The first real run also verifies the Qwen tool-call parser against the model's actual
output format (the one piece CI cannot cover) — sanity-check `eval_report.md`'s tool-call
column is non-zero on a few `explain`/`counterfactual` cases.
