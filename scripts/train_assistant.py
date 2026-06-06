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
    p.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="max checkpoints to keep (best + most recent). Use 0 to keep every epoch "
        "(e.g. for an epoch sweep).",
    )
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


def main() -> None:
    args = parse_args()
    settings = _resolve_settings(args)

    # Lazy heavy imports - keep this module importable without a GPU.
    # Unsloth MUST be imported before trl/transformers/peft so its kernel +
    # memory patches apply (it warns and degrades VRAM use otherwise). This
    # ordering is load-bearing, so the block opts out of import sorting (I001).
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from tradinglib.training.data import prepare_records

    train_records, n_dropped = prepare_records(args.train, require_nonempty=True)
    if n_dropped:
        print(f"WARNING: dropped {n_dropped} invalid training example(s)")
    train_ds = Dataset.from_list(train_records)
    eval_ds = None
    if args.eval:
        eval_records, _ = prepare_records(args.eval)
        eval_ds = Dataset.from_list(eval_records)

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

    # Render conversational rows to a flat ``text`` column via the Qwen chat
    # template (tool_calls/tool turns included). Unsloth's SFTTrainer does NOT
    # auto-apply the chat template to a `messages` column -- it requires `text`,
    # a prompt/completion pair, or a formatting_func -- so we materialize it here.
    def _to_text(rec: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(rec["messages"], tokenize=False)}

    train_ds = train_ds.map(_to_text, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(_to_text, remove_columns=eval_ds.column_names)

    sft_config = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.grad_accum_steps,
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        max_steps=(args.max_steps if args.max_steps is not None else -1),
        seed=settings.seed,
        logging_steps=1,
        # TRL >=0.20 renamed SFTConfig.max_seq_length -> max_length (verified on trl 0.24).
        max_length=settings.max_seq_len,
        # Pin the EOS explicitly: under Unsloth+trl 0.24 a None eos_token resolves to
        # an unresolved '<EOS_TOKEN>' sentinel that fails TRL's vocab check. The Qwen
        # tokenizer's real EOS is '<|im_end|>'.
        eos_token=tokenizer.eos_token,
        # Keep the best (lowest eval_loss) checkpoint, not the final epoch: on small
        # data the model overfits after a few epochs (eval_loss turns up and judged
        # answer quality collapses), so the last epoch can be a worse model.
        # CAVEAT: eval_loss is only a proxy and can UNDER-select for tool-call
        # accuracy -- the eval_loss minimum (often epoch ~2) can sit an epoch
        # before tool-call formatting fully locks in (epoch ~3), where eval_loss
        # is barely higher. For a model that ships on the eval gate, prefer an
        # epoch sweep: train with `--save-total-limit 0` (keep every epoch) and
        # pick the checkpoint via scripts/score_local_noapi.py. See docs.
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if eval_ds is not None else "no",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 0 -> keep every epoch checkpoint (None disables the cap in TRL/transformers).
        save_total_limit=(None if args.save_total_limit == 0 else args.save_total_limit),
    )
    trainer = SFTTrainer(
        model=model,
        # TRL >=0.13 replaced the `tokenizer=` arg with `processing_class=` (verified on trl 0.24).
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )
    # Assistant-only loss: mask the system prompt, user turns, and tool outputs so
    # loss is computed only on the assistant's tool-calls + final answers (ChatML
    # markers). Focuses limited training signal on what the ship-bar measures.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
