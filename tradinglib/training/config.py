"""QLoRA + SFT hyperparameter settings (plain dataclasses, no heavy deps).

Defaults are tuned for a single 16 GB consumer GPU (RTX 5080) fine-tuning
Qwen2.5-7B in 4-bit. The training script reads these; tests pin the defaults so
an accidental change is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _qwen2_target_modules() -> tuple[str, ...]:
    return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class LoRASettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=_qwen2_target_modules)
    bias: str = "none"


@dataclass(frozen=True)
class TrainSettings:
    # Unsloth ships pre-quantized 4-bit bases; this id loads fastest. If using the
    # TRL/bnb fallback, swap to "Qwen/Qwen2.5-7B-Instruct".
    base_model: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_len: int = 2048
    epochs: int = 2
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    seed: int = 42
    lora: LoRASettings = field(default_factory=LoRASettings)
