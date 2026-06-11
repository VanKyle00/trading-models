"""Tests for per-tier LLM scoring sanitization and mechanical metrics."""

from __future__ import annotations

import json

import pandas as pd

from tradinglib.assistant.provider import StubProvider
from tradinglib.assistant.types import AssistantTurn, Usage
from tradinglib.sentiment import scoring


def _turn(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(100, 100))


def _provider(payload: dict | str) -> StubProvider:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return StubProvider([_turn(text)])


def test_score_tier_valid_json_with_noise() -> None:
    payload = {
        "score": 0.7,
        "stance": "bullish",
        "confidence": 0.8,
        "summary": "Upbeat coverage.",
        "key_themes": ["dc demand"],
        "evidence_indices": [0, 2],
    }
    out = scoring.score_tier(
        _provider("noise " + json.dumps(payload) + " trailing"), "official", "pack"
    )
    assert out["score"] == 0.7 and out["stance"] == "bullish"
    assert out["evidence_indices"] == [0, 2] and out["parse_error"] is False


def test_score_tier_clamps_and_validates() -> None:
    payload = {
        "score": 3.2,
        "stance": "moonshot",
        "confidence": -2,
        "summary": "x",
        "key_themes": [1, 2, 3, 4, 5, 6, 7],
        "evidence_indices": [0, 0, True, -1, "2", 1, 3, 4, 9],
    }
    out = scoring.score_tier(_provider(payload), "viral", "pack")
    assert out["score"] == 1.0 and out["stance"] is None and out["confidence"] == 0.0
    assert len(out["key_themes"]) == 5
    assert out["evidence_indices"] == [
        0,
        1,
        3,
        4,
        9,
    ]  # bools/dupes/negatives/strings dropped, capped at 5


def test_score_tier_malformed_json_degrades() -> None:
    out = scoring.score_tier(_provider("not json at all { broken"), "forums", "pack")
    assert out["score"] is None and out["parse_error"] is True
    assert "not json" in out["summary"]


def test_score_tier_provider_exception_degrades() -> None:
    class _Boom:
        def complete(self, system, conversation, tools):
            raise RuntimeError("api down")

    out = scoring.score_tier(_Boom(), "official", "pack")
    assert out["score"] is None and out["parse_error"] is True
    assert "unavailable" in out["summary"]


def test_empty_scoring_shell() -> None:
    out = scoring.empty_scoring()
    assert out["score"] is None and out["summary"] == "" and out["parse_error"] is False


def _reddit_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df):
        df["created"] = pd.to_datetime(df["created"], utc=True)
    return df


def test_forum_metrics() -> None:
    sa = pd.DataFrame({"title": ["a", "b"]})
    reddit = _reddit_df(
        [
            {"created": "2026-06-10", "score": 10, "num_comments": 4},
            {"created": "2026-06-09", "score": 30, "num_comments": 6},
        ]
    )
    m = scoring.forum_metrics(sa, reddit)
    assert m == {"post_count": 4, "mean_upvotes": 20.0, "mean_comments": 5.0}
    m2 = scoring.forum_metrics(sa, pd.DataFrame())
    assert m2 == {"post_count": 2, "mean_upvotes": None, "mean_comments": None}


def test_viral_metrics_ratio_and_spike() -> None:
    st = pd.DataFrame({"sentiment": ["Bullish", "Bullish", "Bearish", None]})
    wsb = pd.DataFrame({"title": ["a"]})
    idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=97, freq="D")
    interest = pd.Series([10.0] * 90 + [30.0] * 7, index=idx)
    bsky = pd.DataFrame({"text": ["a", "b", "c"]})
    m = scoring.viral_metrics(wsb, st, bsky, interest)
    assert m["wsb_mentions"] == 1 and m["st_messages"] == 4
    assert m["st_bullish"] == 2 and m["st_bearish"] == 1
    assert m["st_bull_bear_ratio"] == 2.0
    assert m["trends_spike"] is not None and m["trends_spike"] > 2.0
    assert m["bsky_mentions"] == 3


def test_viral_metrics_guards() -> None:
    st = pd.DataFrame({"sentiment": ["Bullish"]})
    m = scoring.viral_metrics(pd.DataFrame(), st, pd.DataFrame(), None)
    assert m["st_bull_bear_ratio"] is None  # no bears -> undefined, not inf
    assert m["trends_spike"] is None
    assert m["bsky_mentions"] == 0


def test_trends_spike_zero_baseline_is_none() -> None:
    idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=20, freq="D")
    interest = pd.Series([0.0] * 13 + [50.0] * 7, index=idx)
    assert scoring.trends_spike(interest) is None


def test_official_metrics() -> None:
    assert scoring.official_metrics(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1, 2]})) == {
        "headline_count": 3
    }
