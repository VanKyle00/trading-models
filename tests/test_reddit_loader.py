"""Tests for the Reddit search loader (praw stubbed at the client boundary)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _submission(
    title: str, *, score: int = 100, comments: int = 20, selftext: str = ""
) -> SimpleNamespace:
    return SimpleNamespace(
        created_utc=1_781_100_000.0,  # 2026-06-10-ish, UTC epoch seconds
        title=title,
        selftext=selftext,
        score=score,
        num_comments=comments,
        permalink=f"/r/sub/comments/abc/{title[:8].lower().replace(' ', '_')}/",
    )


class _FakeReddit:
    def __init__(self, by_sub: dict[str, list]) -> None:
        self.by_sub = by_sub
        self.queries: list = []

    def subreddit(self, name: str):
        outer = self

        class _Sub:
            def search(self, query: str, *, sort: str, time_filter: str, limit: int):
                outer.queries.append((name, query, sort, time_filter, limit))
                return iter(outer.by_sub.get(name, []))

        return _Sub()


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.forums import reddit as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_reddit_missing_credentials_raises(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    with pytest.raises(loader.MissingRedditCredentials):
        loader.get_reddit_posts("NVDA", ("stocks",))


def test_reddit_schema_concat_and_query(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeReddit(
        {
            "stocks": [_submission("NVDA is undervalued", selftext="long thesis here")],
            "investing": [_submission("Trimming NVDA", score=50, comments=5)],
        }
    )
    monkeypatch.setattr(loader, "_make_reddit", lambda: fake)
    df = loader.get_reddit_posts("NVDA", ("stocks", "investing"))
    assert list(df.columns) == [
        "ticker",
        "subreddit",
        "created",
        "title",
        "text",
        "score",
        "num_comments",
        "url",
    ]
    assert len(df) == 2
    assert set(df["subreddit"]) == {"stocks", "investing"}
    assert df.iloc[0]["url"].startswith("https://www.reddit.com/r/")
    assert str(df["created"].dt.tz) == "UTC"
    _sub, query, _sort, time_filter, _limit = fake.queries[0]
    assert query == "NVDA OR $NVDA"
    assert time_filter == "week"


def test_reddit_cached_needs_no_credentials(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeReddit({"stocks": [_submission("NVDA earnings play")]})
    monkeypatch.setattr(loader, "_make_reddit", lambda: fake)
    loader.get_reddit_posts("NVDA", ("stocks",))  # populates the day's cache

    def _boom() -> None:
        raise loader.MissingRedditCredentials("no creds")

    monkeypatch.setattr(loader, "_make_reddit", _boom)
    df = loader.get_reddit_posts("NVDA", ("stocks",))  # cache hit — no client needed
    assert len(df) == 1


def test_reddit_empty_results_ok(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_make_reddit", lambda: _FakeReddit({}))
    df = loader.get_reddit_posts("ZZZZ", ("stocks",))
    assert df.empty
    assert list(df.columns) == [
        "ticker",
        "subreddit",
        "created",
        "title",
        "text",
        "score",
        "num_comments",
        "url",
    ]
