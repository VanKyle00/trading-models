"""Filesystem layout helpers — discover the repo root and the data
directories without hard-coded paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Walk up from this file until we find ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate repo root — no pyproject.toml in any parent")


def raw_dir(source: str) -> Path:
    return repo_root() / "data" / "raw" / source


def processed_dir(source: str) -> Path:
    return repo_root() / "data" / "processed" / source
