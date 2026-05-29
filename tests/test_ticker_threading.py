"""Locked models must reject a foreign ticker before touching the network."""

from __future__ import annotations

import pytest

from app.adapters import run
from app.models_registry import list_models


def _model(family: str) -> dict:
    return next(m for m in list_models() if m["family"] == family)


@pytest.mark.parametrize("family", ["ml", "microstructure", "alt-data"])
def test_locked_model_rejects_foreign_ticker(family: str) -> None:
    model = _model(family)
    with pytest.raises(ValueError, match="locked"):
        run(model, "2023-01-01", "2024-01-01", symbol="NVDA")
