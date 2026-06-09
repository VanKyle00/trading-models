"""Stage 4a: blend FA, setup, and qualitative scores into a final ranking.

``final_score = 0.35*fa + 0.45*setup + 0.20*(qualitative/10)``; when no
qualitative score exists (``--skip-llm`` or a failed brief) the remaining
weights are renormalized. A ticker whose LLM brief says ``stance: avoid`` or
carries red flags is pinned to the bottom of the list with the reason shown
— never silently dropped.
"""

from __future__ import annotations

_W_FA = 0.35
_W_SETUP = 0.45
_W_QUAL = 0.20


def _final_score(candidate: dict) -> float:
    fa = float(candidate.get("fa_score") or 0.0)
    setup = float(candidate.get("setup_score") or 0.0)
    qual = candidate.get("qualitative_score")
    if qual is None:
        return (_W_FA * fa + _W_SETUP * setup) / (_W_FA + _W_SETUP)
    return _W_FA * fa + _W_SETUP * setup + _W_QUAL * (float(qual) / 10.0)


def _pin_reason(candidate: dict) -> str | None:
    if candidate.get("stance") == "avoid":
        return "LLM brief stance: avoid"
    red_flags = candidate.get("red_flags") or []
    if red_flags:
        return "red flags: " + "; ".join(str(f) for f in red_flags)
    return None


def rank_candidates(candidates: list[dict], *, top: int | None = None) -> list[dict]:
    """Return candidates sorted best-first, pinned names last, truncated to ``top``."""
    ranked = []
    for candidate in candidates:
        out = dict(candidate)
        out["final_score"] = _final_score(candidate)
        reason = _pin_reason(candidate)
        out["pinned"] = reason is not None
        out["pinned_reason"] = reason or ""
        ranked.append(out)
    ranked.sort(key=lambda c: (c["pinned"], -c["final_score"]))
    return ranked[:top] if top is not None else ranked
