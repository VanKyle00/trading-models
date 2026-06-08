"""Grid search over a parameter space with an honest trial count.

``grid_search`` evaluates every point in the Cartesian product of ``param_grid``
with a caller-supplied ``score_fn`` and returns the best configuration plus the
number of configurations tried (``n_trials``) — the value to feed the Deflated
Sharpe so the multiple-testing penalty reflects reality.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    best_params: dict
    best_score: float
    n_trials: int
    results: list[tuple[dict, float]] = field(default_factory=list)


def expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a param grid -> list of param dicts (deterministic)."""
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(param_grid[k] for k in keys))
    ]


def grid_search(param_grid: dict[str, list], score_fn: Callable[[dict], float]) -> SearchResult:
    """Evaluate every config; return the highest-scoring one and the trial count."""
    configs = expand_grid(param_grid)
    results = [(cfg, float(score_fn(cfg))) for cfg in configs]
    best_params, best_score = max(results, key=lambda pair: pair[1])
    return SearchResult(
        best_params=best_params, best_score=best_score, n_trials=len(configs), results=results
    )
