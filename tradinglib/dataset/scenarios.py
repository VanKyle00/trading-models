"""Scenario generation: model x legal ticker x question template."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from tradinglib.dataset.templates import (
    DEFAULT_WINDOW,
    QUESTION_TEMPLATES,
    TICKER_BASKET,
    WINDOWS,
)
from tradinglib.service import list_specs


@dataclass(frozen=True)
class Scenario:
    category: str
    question: str
    model_id: str  # "" for model-agnostic refusals
    symbol: str | None  # None -> tool uses model default
    start: str
    end: str
    params: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)  # e.g. {"fee_bps": 10}


def _symbols_for(spec, rng: random.Random, n: int) -> list[str | None]:
    """Legal symbols to exercise for a model, respecting its ticker mode."""
    if spec.ticker_mode == "free":
        return rng.sample(TICKER_BASKET, min(n, len(TICKER_BASKET)))
    choices = list(spec.ticker_choices) or [spec.default_ticker]
    return [rng.choice(choices) for _ in range(n)]


def generate_scenarios(seed: int = 0, per_model_per_category: int = 3) -> list[Scenario]:
    """Enumerate scenarios across models x legal tickers x templates.

    Deterministic for a given seed. ``explain``/``counterfactual``/``methodology``
    are generated per model; ``refusal`` is generated once (model-agnostic) using
    the basket so the assistant learns to decline regardless of context.
    """
    rng = random.Random(seed)
    specs = list_specs()
    out: list[Scenario] = []

    for spec in specs:
        window = WINDOWS.get(spec.family, DEFAULT_WINDOW)
        start, end = window
        for category in ("explain", "counterfactual", "methodology"):
            templates = QUESTION_TEMPLATES[category]
            syms = _symbols_for(spec, rng, per_model_per_category)
            for i in range(per_model_per_category):
                template = templates[(i + rng.randrange(len(templates))) % len(templates)]
                symbol = syms[i % len(syms)]
                symbol2 = rng.choice([s for s in TICKER_BASKET if s != symbol])
                overrides = {"fee_bps": 10.0} if "fees are 10 bps" in template else {}
                question = template.format(
                    model_name=spec.name,
                    symbol=symbol or spec.default_ticker,
                    symbol2=symbol2,
                    start=start,
                    end=end,
                )
                out.append(
                    Scenario(
                        category=category,
                        question=question,
                        model_id=spec.id,
                        symbol=symbol,
                        start=start,
                        end=end,
                        overrides=overrides,
                    )
                )

    # refusals: model-agnostic, generated once
    for i in range(per_model_per_category * 2):
        template = QUESTION_TEMPLATES["refusal"][i % len(QUESTION_TEMPLATES["refusal"])]
        symbol = rng.choice(TICKER_BASKET)
        question = template.format(
            model_name="our portfolio",
            symbol=symbol,
            symbol2=symbol,
            start=DEFAULT_WINDOW[0],
            end=DEFAULT_WINDOW[1],
        )
        out.append(
            Scenario(
                category="refusal",
                question=question,
                model_id="",
                symbol=None,
                start=DEFAULT_WINDOW[0],
                end=DEFAULT_WINDOW[1],
            )
        )

    rng.shuffle(out)
    return out
