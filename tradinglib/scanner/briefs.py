"""Stage 3: LLM document briefs for the setup finalists.

For each finalist we assemble a bounded "doc pack" — the latest 8-K excerpt,
the latest 10-Q/10-K MD&A opening, recent news headlines, the FA metrics and
the detected setup — and make ONE provider call demanding strict JSON:

    {thesis, catalysts, risks, red_flags, stance, qualitative_score, confidence}

The provider is injected (``ClaudeProvider`` from the CLI, ``StubProvider``
in tests, the user's own model later). Malformed JSON degrades gracefully:
the raw text is kept as the thesis and the qualitative score stays ``None``
so ranking falls back to FA + setup weights.
"""

from __future__ import annotations

import json

from tradinglib.assistant.provider import LLMProvider
from tradinglib.assistant.types import UserMsg
from tradinglib.loaders.filings.edgar import get_filing_text, get_recent_filings
from tradinglib.loaders.news.yfinance import get_news

_PACK_MAX_CHARS = 30_000
_8K_MAX_CHARS = 4_000
_10X_MAX_CHARS = 3_000
_NEWS_MAX_ITEMS = 15
_MDA_MARKER = "discussion and analysis"
_STANCES = ("favorable", "neutral", "avoid")

BRIEF_SYSTEM_PROMPT = (
    "You are an equity research analyst. You receive a document pack for one "
    "stock: recent SEC filing excerpts, news headlines, fundamental metrics, "
    "and a detected technical setup for a 2-week-to-6-month swing trade. "
    "Assess whether the fundamentals and news SUPPORT or UNDERMINE the trade. "
    "Respond with ONLY a JSON object, no prose before or after, with exactly "
    "these keys: thesis (string, 2-3 sentences), catalysts (array of strings), "
    "risks (array of strings), red_flags (array of strings — ONLY serious "
    "issues like restatements, going-concern doubt, investigations, guidance "
    "withdrawals), stance (one of: favorable, neutral, avoid), "
    "qualitative_score (number 0-10), confidence (number 0-1). "
    "Ground every claim in the documents provided; do not invent facts."
)


def _excerpt(text: str, *, marker: str | None, max_chars: int) -> str:
    """Slice from the first occurrence of ``marker`` (case-insensitive) if found."""
    if marker is not None:
        pos = text.lower().find(marker)
        if pos > 0:
            text = text[pos - 20 if pos >= 20 else 0 :]
    return text[:max_chars]


def build_doc_pack(candidate: dict, cik: int, *, refresh: bool = False) -> str:
    """Assemble the bounded document pack for one finalist."""
    ticker = candidate["ticker"]
    parts: list[str] = [
        f"# {ticker} — {candidate.get('name', '')} ({candidate.get('sector', '?')})",
        "",
        "## Detected setup",
        json.dumps(candidate.get("setups", []), default=str),
        f"FA score (cross-sectional percentile composite): {candidate.get('fa_score')}",
        "",
    ]

    filings = get_recent_filings(cik, refresh=refresh)
    for form_group, marker, cap, label in (
        (("8-K",), None, _8K_MAX_CHARS, "Latest 8-K excerpt"),
        (("10-Q", "10-K"), _MDA_MARKER, _10X_MAX_CHARS, "Latest 10-Q/10-K MD&A excerpt"),
    ):
        subset = filings[filings["form"].isin(form_group)]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        text = get_filing_text(cik, row["accession"], row["primary_document"])
        parts.extend(
            [
                f"## {label} ({row['form']}, filed {str(row['filing_date'])[:10]})",
                _excerpt(text, marker=marker, max_chars=cap),
                "",
            ]
        )

    news = get_news(ticker, max_items=_NEWS_MAX_ITEMS, refresh=refresh)
    if len(news) > 0:
        parts.append("## Recent news headlines")
        for item in news.itertuples():
            parts.append(
                f"- [{str(item.published)[:10]}] {item.title} ({item.publisher}): {item.summary}"
            )
        parts.append("")

    return "\n".join(parts)[:_PACK_MAX_CHARS]


def _sanitize(raw: dict, *, parse_error: bool = False) -> dict:
    stance = raw.get("stance")
    if stance not in _STANCES:
        stance = None
    score = raw.get("qualitative_score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None and not 0.0 <= score <= 10.0:
        score = None
    return {
        "thesis": str(raw.get("thesis", "")),
        "catalysts": [str(x) for x in raw.get("catalysts") or []],
        "risks": [str(x) for x in raw.get("risks") or []],
        "red_flags": [str(x) for x in raw.get("red_flags") or []],
        "stance": stance,
        "qualitative_score": score,
        "confidence": raw.get("confidence"),
        "parse_error": parse_error,
    }


def write_brief(provider: LLMProvider, pack: str) -> dict:
    """One provider call -> sanitized brief dict; never raises on bad JSON."""
    turn = provider.complete(BRIEF_SYSTEM_PROMPT, [UserMsg(pack)], tools=[])
    text = turn.text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return _sanitize(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    return _sanitize({"thesis": text}, parse_error=True)


def brief_candidates(
    provider: LLMProvider,
    candidates: list[dict],
    *,
    ciks: dict[str, int],
    errors: list[dict],
    refresh: bool = False,
) -> None:
    """Brief every candidate in place; per-ticker failures land in ``errors``."""
    for candidate in candidates:
        ticker = candidate["ticker"]
        try:
            cik = ciks[ticker]
            pack = build_doc_pack(candidate, cik, refresh=refresh)
            brief = write_brief(provider, pack)
            candidate["brief"] = brief
            candidate["qualitative_score"] = brief["qualitative_score"]
            candidate["stance"] = brief["stance"]
            candidate["red_flags"] = brief["red_flags"]
        except Exception as exc:
            errors.append({"ticker": ticker, "stage": "brief", "error": str(exc)})
