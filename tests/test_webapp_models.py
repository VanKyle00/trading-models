"""Tests for the /models page (strategy registry + standalone model specs)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tradinglib.service import list_specs
from tradinglib.tournament.strategies import STRATEGIES
from webapp.main import create_app


def test_models_page_lists_every_tournament_strategy() -> None:
    html = TestClient(create_app()).get("/models").text

    for sdef in STRATEGIES.values():
        assert sdef.name in html  # generated from the registry, never hand-listed
        assert sdef.description in html
    assert "mean reversion" in html  # style chips render human-readably


def test_models_page_shows_param_grids_and_stances() -> None:
    html = TestClient(create_app()).get("/models").text

    grid = STRATEGIES["sma_cross"].param_grid
    for param in grid:
        assert param in html
    assert "long + short" in html
    assert "entry / stop / target" in html


def test_models_page_lists_standalone_models_with_links() -> None:
    html = TestClient(create_app()).get("/models").text

    for spec in list_specs():
        assert spec.name in html
        assert spec.id in html  # link target carries the repo path


def test_index_links_to_models() -> None:
    assert 'href="/models"' in TestClient(create_app()).get("/").text
