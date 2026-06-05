import tomllib
from pathlib import Path


def _opt_deps():
    data = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    return data["project"]["optional-dependencies"]


def test_train_group_exists_with_core_packages():
    train = _opt_deps()["train"]
    joined = " ".join(train).lower()
    for pkg in ("transformers", "trl", "peft", "datasets", "accelerate"):
        assert pkg in joined, f"missing {pkg} in [train]"


def test_base_install_does_not_pull_torch():
    data = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    base = " ".join(data["project"]["dependencies"]).lower()
    assert "torch" not in base
