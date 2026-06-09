"""Tests for the LLM document-brief stage (Stage 3 of the scanner)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, Usage
from tradinglib.scanner import briefs


def _turn(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(100, 100))


def _candidate() -> dict:
    return {
        "ticker": "AAPL",
        "name": "Apple",
        "sector": "Tech",
        "fa_score": 0.9,
        "setup_score": 0.7,
        "setups": [
            {
                "setup_type": "base_breakout",
                "score": 0.7,
                "trigger_level": 210.0,
                "stop_level": 195.0,
                "evidence": {"atr": 2.0},
            }
        ],
        "qualitative_score": None,
        "stance": None,
        "red_flags": [],
        "brief": None,
        "earnings_warning": False,
    }


@pytest.fixture
def patched_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_filings(
        cik, *, forms=("8-K", "10-Q", "10-K"), days=120, asof=None, refresh=False, client=None
    ):
        return pd.DataFrame(
            {
                "cik": [cik, cik],
                "form": ["8-K", "10-Q"],
                "accession": ["acc-8k", "acc-10q"],
                "filing_date": pd.to_datetime(["2026-06-01", "2026-05-02"], utc=True),
                "primary_document": ["pr.htm", "q.htm"],
            }
        )

    def fake_text(cik, accession, primary_document, *, max_chars=20_000, client=None):
        if accession == "acc-8k":
            return "Apple reported record revenue and raised guidance." * 5
        return "Management's Discussion and Analysis: services margin expanded." * 5

    def fake_news(ticker, *, max_items=15, refresh=False):
        return pd.DataFrame(
            {
                "ticker": [ticker],
                "published": pd.to_datetime(["2026-06-05"], utc=True),
                "title": ["Apple unveils new product"],
                "summary": ["A big launch."],
                "publisher": ["Reuters"],
            }
        )

    monkeypatch.setattr(briefs, "get_recent_filings", fake_filings)
    monkeypatch.setattr(briefs, "get_filing_text", fake_text)
    monkeypatch.setattr(briefs, "get_news", fake_news)


def test_doc_pack_contains_all_documents(patched_loaders) -> None:
    pack = briefs.build_doc_pack(_candidate(), cik=320193)

    assert "AAPL" in pack
    assert "record revenue" in pack  # 8-K excerpt
    assert "Discussion and Analysis" in pack  # 10-Q excerpt
    assert "Apple unveils new product" in pack  # news headline
    assert "base_breakout" in pack  # setup context
    assert len(pack) <= briefs._PACK_MAX_CHARS


def test_write_brief_parses_strict_json(patched_loaders) -> None:
    payload = {
        "thesis": "Strong product cycle.",
        "catalysts": ["WWDC"],
        "risks": ["China demand"],
        "red_flags": [],
        "stance": "favorable",
        "qualitative_score": 8,
        "confidence": 0.7,
    }
    provider = StubProvider([_turn(json.dumps(payload))])

    brief = briefs.write_brief(provider, "doc pack text")

    assert brief["stance"] == "favorable"
    assert brief["qualitative_score"] == 8.0
    assert brief["catalysts"] == ["WWDC"]
    assert brief["parse_error"] is False


def test_write_brief_extracts_json_from_prose() -> None:
    text = 'Here is my analysis:\n{"thesis": "ok", "stance": "neutral", "qualitative_score": 5}'
    provider = StubProvider([_turn(text)])

    brief = briefs.write_brief(provider, "pack")

    assert brief["stance"] == "neutral"
    assert brief["qualitative_score"] == 5.0


def test_write_brief_degrades_on_malformed_json() -> None:
    provider = StubProvider([_turn("I cannot produce JSON today, sorry.")])

    brief = briefs.write_brief(provider, "pack")

    assert brief["parse_error"] is True
    assert brief["qualitative_score"] is None
    assert brief["stance"] is None
    assert "cannot produce JSON" in brief["thesis"]


def test_write_brief_sanitizes_out_of_range_values() -> None:
    payload = {"thesis": "x", "stance": "buy now!!", "qualitative_score": 47}
    provider = StubProvider([_turn(json.dumps(payload))])

    brief = briefs.write_brief(provider, "pack")

    assert brief["stance"] is None  # not one of favorable|neutral|avoid
    assert brief["qualitative_score"] is None  # outside 0-10


def test_brief_candidates_mutates_and_isolates_errors(patched_loaders) -> None:
    good = _candidate()
    bad = _candidate()
    bad["ticker"] = "BAD"

    payload = {
        "thesis": "ok",
        "catalysts": [],
        "risks": [],
        "red_flags": ["accounting restatement"],
        "stance": "avoid",
        "qualitative_score": 2,
        "confidence": 0.9,
    }
    provider = StubProvider([_turn(json.dumps(payload)), _turn(json.dumps(payload))])

    errors: list[dict] = []
    briefs.brief_candidates(
        provider, [good, bad], ciks={"AAPL": 320193}, errors=errors
    )  # BAD has no CIK -> error recorded, scan continues

    assert good["stance"] == "avoid"
    assert good["qualitative_score"] == 2.0
    assert good["red_flags"] == ["accounting restatement"]
    assert good["brief"]["thesis"] == "ok"
    assert len(errors) == 1
    assert errors[0]["ticker"] == "BAD"
    assert errors[0]["stage"] == "brief"
