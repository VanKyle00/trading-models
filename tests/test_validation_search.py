"""Tests for grid search."""
from __future__ import annotations

from tradinglib.validation.search import SearchResult, expand_grid, grid_search


def test_expand_grid_cartesian_product() -> None:
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = expand_grid(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos and {"a": 2, "b": 30} in combos


def test_expand_grid_empty_is_single_empty_config() -> None:
    assert expand_grid({}) == [{}]


def test_grid_search_picks_max_and_counts_trials() -> None:
    grid = {"x": [1, 2, 3, 4]}
    res = grid_search(grid, score_fn=lambda p: -((p["x"] - 3) ** 2))  # peak at x=3
    assert isinstance(res, SearchResult)
    assert res.best_params == {"x": 3}
    assert res.n_trials == 4
    assert len(res.results) == 4
