"""Scenario generation: model x legal ticker x question template."""

from __future__ import annotations

import random
from collections import Counter
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


def partition_by_ticker(
    scenarios: list[Scenario], eval_frac: float = 0.15, seed: int = 0
) -> tuple[list[Scenario], list[Scenario]]:
    """Strict held-out-by-ticker split: every scenario for a given symbol lands
    entirely in train OR eval, so no ticker (and its templated phrasings) leaks
    train->eval. Refusals (``symbol is None``) are model-agnostic and carry no
    ticker to leak; they're split by a seeded shuffle so eval still exercises
    them. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    symbols = sorted({s.symbol for s in scenarios if s.symbol is not None})
    rng.shuffle(symbols)
    eval_symbols = set(symbols[: int(len(symbols) * eval_frac)])

    ticker_train = [s for s in scenarios if s.symbol is not None and s.symbol not in eval_symbols]
    ticker_eval = [s for s in scenarios if s.symbol is not None and s.symbol in eval_symbols]

    refusals = [s for s in scenarios if s.symbol is None]
    rng.shuffle(refusals)
    n_ref_eval = int(len(refusals) * eval_frac)

    return ticker_train + refusals[n_ref_eval:], ticker_eval + refusals[:n_ref_eval]


# Categories whose answers cite real backtest numbers -> hold these out BY TICKER
# so a ticker's metrics can't leak train->eval. Others (methodology=doc-based,
# refusal=model-agnostic) carry no ticker numbers, so a random within-category
# holdout is leakage-free and gives us guaranteed category coverage in eval.
_TICKER_CATS = ("explain", "counterfactual")


def partition_stratified(
    scenarios: list[Scenario], eval_frac: float = 0.2, seed: int = 0
) -> tuple[list[Scenario], list[Scenario]]:
    """Category-balanced, leakage-aware held-out split.

    explain/counterfactual are held out by ticker, choosing the *rarest* tickers
    first so dominant tickers (e.g. SPY/BTC) stay in training while backtest
    numbers still can't leak. methodology/refusal are held out at random within
    each category (>=1 each) so every category is represented in eval. The two
    aims -- no number leakage where it matters, full category coverage for a
    trustworthy gate -- are why a single rule can't serve both. Deterministic.
    """
    rng = random.Random(seed)

    # 1. hold out the rarest explain/counterfactual tickers up to ~eval_frac.
    tk_counts: Counter[str] = Counter(
        s.symbol for s in scenarios if s.symbol and s.category in _TICKER_CATS
    )
    target = sum(tk_counts.values()) * eval_frac
    held_tickers: set[str] = set()
    held = 0
    for tk, cnt in sorted(tk_counts.items(), key=lambda kv: (kv[1], kv[0])):
        if held >= target:
            break
        held_tickers.add(tk)
        held += cnt

    train: list[Scenario] = []
    eval_: list[Scenario] = []

    # 2. ticker-bound categories: eval iff the ticker is held out.
    for s in scenarios:
        if s.category in _TICKER_CATS:
            (eval_ if s.symbol in held_tickers else train).append(s)

    # 3. every other category: random within-category holdout, >=1 in eval.
    other_cats = {s.category for s in scenarios} - set(_TICKER_CATS)
    for cat in sorted(other_cats):
        group = [s for s in scenarios if s.category == cat]
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * eval_frac)) if group else 0
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])

    return train, eval_
