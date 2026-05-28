"""Model discovery — parse the YAML frontmatter on every ``model.md``.

This is the single source of truth for "what models exist in this repo".
Both ``scripts/regenerate_models_index.py`` (the MODELS.md generator)
and the Streamlit app under ``app/`` import from here so the index is
defined exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tradinglib.data.paths import repo_root


def parse_frontmatter(path: Path) -> dict[str, Any] | None:
    """Return the YAML frontmatter block at the top of a model.md file, or None."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    data = yaml.safe_load(text[4:end])
    return data if isinstance(data, dict) else None


def find_models(models_root: Path | None = None) -> list[dict[str, Any]]:
    """Scan ``<repo_root>/models`` (or a provided directory) for every
    ``model.md`` and return one dict per model with its frontmatter plus a
    ``_path`` key pointing to the model directory relative to the repo
    root.
    """
    root = models_root if models_root is not None else repo_root() / "models"
    rows: list[dict[str, Any]] = []
    for model_md in sorted(root.rglob("model.md")):
        meta = parse_frontmatter(model_md)
        if meta is None:
            continue
        meta["_path"] = model_md.parent.relative_to(repo_root()).as_posix()
        rows.append(meta)
    return rows
