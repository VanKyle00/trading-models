"""Streamlit-friendly wrapper around :func:`tradinglib.models_index.find_models`.

Caches the discovery scan so the sidebar dropdown doesn't re-walk the
filesystem on every interaction.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from tradinglib.models_index import find_models


@lru_cache(maxsize=1)
def list_models() -> tuple[dict[str, Any], ...]:
    """Return all models discovered from ``model.md`` frontmatters.

    Cached for the lifetime of the process. Returns a tuple so that the
    LRU cache stays hashable.
    """
    return tuple(find_models())


def get_model_by_path(path: str) -> dict[str, Any]:
    for m in list_models():
        if m["_path"] == path:
            return m
    raise KeyError(path)
