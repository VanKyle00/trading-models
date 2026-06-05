import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "train_assistant_cli",
    Path(__file__).resolve().parent.parent / "scripts" / "train_assistant.py",
)


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)  # must NOT import torch/unsloth at module load
    return mod


def test_module_loads_without_heavy_deps():
    mod = _load()
    assert hasattr(mod, "main") and hasattr(mod, "parse_args")


def test_parse_args_defaults():
    mod = _load()
    args = mod.parse_args(["--train", "train.jsonl", "--out", "adapter"])
    assert args.train == "train.jsonl" and args.out == "adapter"
    assert args.eval is None and args.max_steps is None


def test_resolve_settings_applies_overrides():
    mod = _load()
    args = mod.parse_args(
        ["--train", "t", "--out", "o", "--epochs", "3", "--lr", "1e-4", "--base-model", "X"]
    )
    settings = mod._resolve_settings(args)
    assert settings.epochs == 3 and settings.learning_rate == 1e-4 and settings.base_model == "X"
