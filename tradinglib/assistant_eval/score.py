"""Deterministic per-case scoring: tool-call accuracy and grounding.

Tool-call accuracy = greedy match of candidate calls to gold calls (same name +
args equal under numeric tolerance / order-independent keys / case-insensitive
strings), scored matched / max(|gold|, |candidate|) so both missing and spurious
calls cost. Free-text args (a search query) match by content-token overlap rather
than exact string. Grounding reuses the existing deterministic verifier unchanged."""

from __future__ import annotations

import re
from typing import Any

from tradinglib.assistant.types import ToolCall
from tradinglib.assistant_eval.runner import RunResult
from tradinglib.dataset.grounding import is_grounded

_REL = 0.02
_ABS = 0.01

# Free-text args (a search query) are scored by content-token overlap, not exact
# string equality: the gold trace's wording is just one of many phrasings that
# retrieve the same docs, so an exact match penalized query *phrasing* rather than
# tool *use* (e.g. gold "...slippage execution model" vs "...slippage fee model",
# or an over-specific gold "Deflated Sharpe ratio definition meaning" vs the
# model's "Deflated Sharpe ratio"). Identifier args (model_id, symbol, ...) are
# NOT free-text and keep exact matching via _value_match.
_FREE_TEXT_KEYS = frozenset({"query"})
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "for",
        "in",
        "on",
        "to",
        "is",
        "are",
        "be",
        "does",
        "do",
        "did",
        "what",
        "how",
        "why",
        "which",
        "this",
        "that",
        "these",
        "those",
        "here",
        "it",
        "its",
        "with",
        "as",
        "at",
        "by",
        "model",
    }
)
# Match when shared content tokens >= differing tokens. Tuned so paraphrases /
# over-specific gold pass while genuinely different queries (different model,
# different question) stay below: e.g. "X data source" vs "X train test split"
# lands at ~0.44, two different models at ~0.25.
_QUERY_MATCH_THRESHOLD = 0.5


def _value_match(x: Any, y: Any) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return x is y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y) <= max(_ABS, _REL * abs(y))
    if isinstance(x, str) and isinstance(y, str):
        return x.strip().lower() == y.strip().lower()
    if isinstance(x, dict) and isinstance(y, dict):
        return _args_match(x, y)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_value_match(a, b) for a, b in zip(x, y, strict=False))
    return x == y


def _tokens(s: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", s.lower())
    toks = {t for t in raw if t not in _QUERY_STOPWORDS}
    return toks or set(raw)  # if stopwords ate everything, fall back to raw tokens


def _text_overlap_match(candidate: str, gold: str) -> bool:
    """Free-text match by content-token Jaccard, with an exact-match fast path."""
    if candidate.strip().lower() == gold.strip().lower():
        return True
    c, g = _tokens(candidate), _tokens(gold)
    if not c or not g:
        return False
    return len(c & g) / len(c | g) >= _QUERY_MATCH_THRESHOLD


def _args_match(candidate: dict, gold: dict) -> bool:
    """The candidate must satisfy every arg the gold trace specified (right keys,
    matching values); extra optional args on the candidate are allowed. The
    minimal gold traces omit defaults the model often spells out (some are inert
    knobs the model doesn't even support), so penalizing that over-specification
    measured tool *selection* unfairly. Wrong or missing gold args still fail.
    Free-text args (see _FREE_TEXT_KEYS) match by token overlap, not exact string."""
    for k in gold:
        if k not in candidate:
            return False
        cv, gv = candidate[k], gold[k]
        if k in _FREE_TEXT_KEYS and isinstance(cv, str) and isinstance(gv, str):
            if not _text_overlap_match(cv, gv):
                return False
        elif not _value_match(cv, gv):
            return False
    return True


def tool_call_score(candidate: list[ToolCall], gold: list[ToolCall]) -> float:
    if not gold:
        return 1.0 if not candidate else 0.0
    remaining = list(gold)
    matched = 0
    for c in candidate:
        for i, g in enumerate(remaining):
            if c.name == g.name and _args_match(c.input, g.input):
                matched += 1
                remaining.pop(i)
                break
    return matched / max(len(gold), len(candidate))


def grounded(run: RunResult) -> bool:
    ok, _missing = is_grounded(run.final_answer, run.tool_outputs)
    return ok
