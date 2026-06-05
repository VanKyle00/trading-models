import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_dataset_cli", Path(__file__).resolve().parent.parent / "scripts" / "build_dataset.py"
)


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def test_parse_args_defaults():
    mod = _load()
    args = mod.parse_args(["--out", "x", "--n", "2"])
    assert args.out == "x" and args.n == 2 and args.seed == 0


def test_limit_truncates_scenarios():
    mod = _load()
    scns = list(range(100))
    assert mod._apply_limit(scns, 5) == list(range(5))
    assert mod._apply_limit(scns, None) == scns
