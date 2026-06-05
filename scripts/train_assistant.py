"""QLoRA fine-tune Qwen2.5-7B on the harness dataset. Run in WSL2 on the RTX 5080.

    uv run python scripts/train_assistant.py --train data/dataset/train.jsonl \
        --eval data/dataset/eval.jsonl --out adapters/qwen25-7b-assistant --max-steps 10

See docs/training-assistant.md for the full runbook (install, smoke, full run).
Heavy deps are imported lazily inside main() so the module stays importable
(and unit-testable) without a GPU.
"""

from __future__ import annotations

import argparse
import dataclasses

from tradinglib.training.config import TrainSettings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune the assistant model.")
    p.add_argument("--train", required=True, help="path to train.jsonl")
    p.add_argument("--out", required=True, help="output dir for the LoRA adapter")
    p.add_argument("--eval", default=None, help="optional eval.jsonl (held-out loss)")
    p.add_argument("--base-model", default=None, help="override base model id")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps for a smoke run")
    return p.parse_args(argv)


def _resolve_settings(args: argparse.Namespace) -> TrainSettings:
    overrides: dict = {}
    if args.base_model is not None:
        overrides["base_model"] = args.base_model
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.max_seq_len is not None:
        overrides["max_seq_len"] = args.max_seq_len
    return dataclasses.replace(TrainSettings(), **overrides)


def validate_dataset_summary(path: str) -> tuple[int, list[str]]:
    """Thin wrapper so the heavy-import-free validator is reachable from main."""
    from tradinglib.training.data import validate_dataset

    return validate_dataset(path)


def main() -> None:
    args = parse_args()
    settings = _resolve_settings(args)

    # Lazy heavy imports - keep this module importable without a GPU.
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from tradinglib.training.data import load_jsonl, to_trl_records, validate_example

    n_ok, problems = validate_dataset_summary(args.train)
    if problems:
        print(f"WARNING: {len(problems)} invalid lines; {n_ok} valid")
    valid = [ex for ex in load_jsonl(args.train) if not validate_example(ex)]
    train_ds = Dataset.from_list(to_trl_records(valid))
    eval_ds = None
    if args.eval:
        eval_ds = Dataset.from_list(to_trl_records(load_jsonl(args.eval)))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.base_model,
        max_seq_length=settings.max_seq_len,
        load_in_4bit=settings.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=settings.lora.r,
        lora_alpha=settings.lora.alpha,
        lora_dropout=settings.lora.dropout,
        target_modules=list(settings.lora.target_modules),
        bias=settings.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=settings.seed,
    )

    sft_config = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.grad_accum_steps,
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        max_steps=args.max_steps or -1,
        seed=settings.seed,
        logging_steps=1,
        eval_strategy="epoch" if eval_ds is not None else "no",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
