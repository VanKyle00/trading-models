"""End-to-end tests for run_sentiment with all loaders stubbed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tradinglib.assistant.types import AssistantTurn, Usage
from tradinglib.sentiment import report as report_mod


def _turn(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, tool_calls=(), stop_reason="end_turn", usage=Usage(100, 100))


class _TierStub:
    """Deterministic provider: routes on the tier label inside the pack header
    (tier LLM calls run concurrently, so scripted-order stubs would race)."""

    def __init__(self, by_label: dict[str, dict]) -> None:
        self.by_label = by_label
        self.calls: list[str] = []

    def complete(self, system, conversation, tools):
        pack = conversation[0].text
        self.calls.append(pack.splitlines()[0])
        for label, payload in self.by_label.items():
            if label in pack:
                return _turn(json.dumps(payload))
        raise AssertionError(f"no stub for pack: {pack[:80]}")


def _scoring(score: float, stance: str) -> dict:
    return {
        "score": score,
        "stance": stance,
        "confidence": 0.8,
        "summary": f"{stance} read.",
        "key_themes": ["theme"],
        "evidence_indices": [0],
    }


_PROVIDER_PAYLOADS = {
    "Official media": _scoring(0.8, "bullish"),
    "Serious forums": _scoring(0.2, "neutral"),
    "Viral / retail": _scoring(-0.4, "bearish"),
}


def _news(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA"] * n,
            "published": pd.to_datetime(["2026-06-10"] * n, utc=True),
            "title": [f"Headline {i}" for i in range(n)],
            "summary": ["s"] * n,
            "publisher": ["Reuters"] * n,
        }
    )


def _gnews(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA"] * n,
            "published": pd.to_datetime(["2026-06-10"] * n, utc=True),
            "title": [f"GN headline {i}" for i in range(n)],
            "publisher": ["CNBC"] * n,
            "url": [f"https://g/{i}" for i in range(n)],
        }
    )


def _sa() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "published": pd.to_datetime(["2026-06-10"], utc=True),
            "title": ["SA: Take Profits"],
            "url": ["https://sa/1"],
        }
    )


def _reddit(sub: str = "stocks") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "subreddit": [sub],
            "created": pd.to_datetime(["2026-06-10"], utc=True),
            "title": [f"{sub} post"],
            "text": ["body"],
            "score": [10],
            "num_comments": [3],
            "url": [f"https://reddit/{sub}"],
        }
    )


def _st() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA"] * 2,
            "created": pd.to_datetime(["2026-06-10", "2026-06-09"], utc=True),
            "body": ["to the moon", "fading this"],
            "sentiment": ["Bullish", "Bearish"],
            "username": ["u1", "u2"],
            "url": ["https://st/1", "https://st/2"],
        }
    )


def _trends() -> pd.Series:
    idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=97, freq="D")
    return pd.Series([10.0] * 90 + [20.0] * 7, index=idx)


@pytest.fixture
def stubbed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(report_mod, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(report_mod, "get_news", lambda t, **kw: _news())
    monkeypatch.setattr(report_mod, "get_google_news", lambda t, **kw: _gnews())
    monkeypatch.setattr(report_mod, "get_seeking_alpha", lambda t, **kw: _sa())
    monkeypatch.setattr(
        report_mod,
        "get_reddit_posts",
        lambda t, subs, **kw: _reddit("wallstreetbets" if "wallstreetbets" in subs else "stocks"),
    )
    monkeypatch.setattr(report_mod, "get_stocktwits", lambda t, **kw: _st())
    monkeypatch.setattr(report_mod, "load_interest", lambda q, **kw: _trends())
    return tmp_path


def test_full_report(stubbed: Path) -> None:
    provider = _TierStub(_PROVIDER_PAYLOADS)
    rep = report_mod.run_sentiment("nvda", provider=provider)
    assert rep.ticker == "NVDA" and rep.status == "ok"
    assert len(rep.tiers) == 3 and len(provider.calls) == 3
    by_tier = {t.tier: t for t in rep.tiers}
    assert by_tier["official"].score == 0.8 and by_tier["official"].status == "ok"
    assert by_tier["viral"].metrics["st_bull_bear_ratio"] == 1.0
    assert rep.overall_bias == round((0.8 + 0.2 - 0.4) / 3, 3)
    assert rep.divergence == {"pair": ["official", "viral"], "gap": 1.2}
    assert by_tier["official"].evidence[0].source in ("Reuters", "CNBC")
    saved = stubbed / "sentiment" / "reports" / "NVDA"
    assert len(list(saved.glob("*.json"))) == 1


def test_same_day_cache_skips_everything(stubbed: Path) -> None:
    provider = _TierStub(_PROVIDER_PAYLOADS)
    first = report_mod.run_sentiment("NVDA", provider=provider)
    second = report_mod.run_sentiment("NVDA", provider=provider)
    assert len(provider.calls) == 3  # not 6 — second call served from the report JSON
    assert second == first


def test_refresh_reruns(stubbed: Path) -> None:
    provider = _TierStub(_PROVIDER_PAYLOADS)
    report_mod.run_sentiment("NVDA", provider=provider)
    report_mod.run_sentiment("NVDA", provider=provider, refresh=True)
    assert len(provider.calls) == 6


def test_source_error_degrades_tier(stubbed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(t, **kw):
        raise RuntimeError("st down")

    monkeypatch.setattr(report_mod, "get_stocktwits", _boom)
    provider = _TierStub(_PROVIDER_PAYLOADS)
    rep = report_mod.run_sentiment("NVDA", provider=provider)
    viral = next(t for t in rep.tiers if t.tier == "viral")
    assert viral.status == "degraded"  # wsb still has content, but a source errored
    assert viral.source_status["stocktwits"].startswith("error")
    assert rep.status == "partial"


def test_all_sources_empty_is_no_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = pd.DataFrame()
    monkeypatch.setattr(report_mod, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(report_mod, "get_news", lambda t, **kw: empty)
    monkeypatch.setattr(report_mod, "get_google_news", lambda t, **kw: empty)
    monkeypatch.setattr(report_mod, "get_seeking_alpha", lambda t, **kw: empty)
    monkeypatch.setattr(report_mod, "get_reddit_posts", lambda t, subs, **kw: empty)
    monkeypatch.setattr(report_mod, "get_stocktwits", lambda t, **kw: empty)
    monkeypatch.setattr(report_mod, "load_interest", lambda q, **kw: pd.Series(dtype=float))

    class _NeverCalled:
        def complete(self, *a, **kw):
            raise AssertionError("LLM must not be called with no content")

    rep = report_mod.run_sentiment("ZZZZ", provider=_NeverCalled())
    assert rep.status == "no_data" and rep.overall_bias is None and rep.divergence is None
    assert all(t.status == "no_data" for t in rep.tiers)


def test_provider_construction_failure_degrades(
    stubbed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_key() -> None:
        raise RuntimeError("ANTHROPIC_API_KEY missing")

    monkeypatch.setattr(report_mod, "ClaudeProvider", _no_key)
    rep = report_mod.run_sentiment("NVDA")  # provider=None -> tries ClaudeProvider()
    assert rep.status == "partial"
    for tier in rep.tiers:
        assert tier.status == "degraded" and "unavailable" in tier.summary
        assert tier.metrics  # mechanical metrics still present


def test_corrupt_cache_self_heals(stubbed: Path) -> None:
    provider = _TierStub(_PROVIDER_PAYLOADS)
    first = report_mod.run_sentiment("NVDA", provider=provider)
    path = next((stubbed / "sentiment" / "reports" / "NVDA").glob("*.json"))
    path.write_text("{ truncated garbage", encoding="utf-8")
    rep = report_mod.run_sentiment("NVDA", provider=provider)  # must refetch, not raise
    assert rep.status == first.status == "ok"
    assert len(provider.calls) == 6  # 3 initial + 3 refetch


def test_trends_failure_does_not_degrade_viral(
    stubbed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(q, **kw):
        raise RuntimeError("429: pytrends mood")

    monkeypatch.setattr(report_mod, "load_interest", _boom)
    rep = report_mod.run_sentiment("NVDA", provider=_TierStub(_PROVIDER_PAYLOADS))
    viral = next(t for t in rep.tiers if t.tier == "viral")
    assert viral.status == "ok"  # wsb + stocktwits content scored fine
    assert viral.source_status["google_trends"].startswith("error")
    assert viral.metrics["trends_spike"] is None
    assert rep.status == "ok"


def test_partial_llm_failure_mixed_statuses(stubbed: Path) -> None:
    class _PartialStub(_TierStub):
        def complete(self, system, conversation, tools):
            if "Viral / retail" in conversation[0].text:
                raise RuntimeError("api hiccup")
            return super().complete(system, conversation, tools)

    rep = report_mod.run_sentiment("NVDA", provider=_PartialStub(_PROVIDER_PAYLOADS))
    by_tier = {t.tier: t for t in rep.tiers}
    assert by_tier["official"].status == "ok" and by_tier["viral"].status == "degraded"
    assert "unavailable" in by_tier["viral"].summary
    assert rep.status == "partial"
    assert rep.overall_bias == round((0.8 + 0.2) / 2, 3)  # mean of the two scored tiers
