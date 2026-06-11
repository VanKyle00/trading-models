"""Tests for bounded tier-pack assembly."""

from __future__ import annotations

import pandas as pd

from tradinglib.sentiment import packs


def _news_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def test_news_items_dedupes_across_sources() -> None:
    yf = _news_df(
        [
            {
                "ticker": "NVDA",
                "published": "2026-06-10",
                "title": "Nvidia rallies on revenue!",
                "summary": "Big quarter.",
                "publisher": "Reuters",
            }
        ]
    )
    gn = _news_df(
        [
            {
                "ticker": "NVDA",
                "published": "2026-06-10",
                "title": "NVIDIA RALLIES ON REVENUE",
                "publisher": "Reuters",
                "url": "https://x",
            },
            {
                "ticker": "NVDA",
                "published": "2026-06-09",
                "title": "A different story",
                "publisher": "CNBC",
                "url": "https://y",
            },
        ]
    )
    items = packs.news_items(yf, gn)
    assert len(items) == 2  # punctuation/case-insensitive title dedupe
    assert items[0]["source"] == "Reuters" and items[0]["url"] == ""


def test_build_pack_indices_align_with_kept_items() -> None:
    items = [
        {
            "source": "Reuters",
            "title": f"t{i}",
            "text": f"text number {i}",
            "url": f"https://e/{i}",
            "published": pd.Timestamp("2026-06-10", tz="UTC"),
        }
        for i in range(5)
    ]
    pack, kept = packs.build_pack("NVDA", "Official media", items, ["headline_count: 5"])
    assert "# NVDA — Official media sentiment pack" in pack
    assert "headline_count: 5" in pack
    for i, item in enumerate(kept):
        assert f"[{i}] " in pack
        assert item["text"] in pack
    lines = pack.splitlines()
    for i in range(len(kept)):
        assert lines[3 + i].startswith(f"[{i}] ")


def test_build_pack_bounds_total_chars() -> None:
    items = [
        {"source": "S", "title": f"t{i}", "text": "x" * 400, "url": "", "published": None}
        for i in range(200)
    ]
    pack, kept = packs.build_pack("NVDA", "Official media", items, [])
    assert len(pack) <= 10_000
    assert 0 < len(kept) < 200  # truncated per item to 280 chars, capped overall


def test_item_text_truncated_and_flattened() -> None:
    items = [
        {"source": "S", "title": "t", "text": "a\nb   c" + "y" * 500, "url": "", "published": None}
    ]
    pack, _kept = packs.build_pack("NVDA", "Viral / retail", items, [])
    line = next(ln for ln in pack.splitlines() if ln.startswith("[0]"))
    assert "\n" not in line and "a b c" in line
    assert len(line) <= 280 + 40  # text cap + prefix slack


def test_age_suffix_present_when_published_known() -> None:
    items = [
        {
            "source": "Reuters",
            "title": "t",
            "text": "hello",
            "url": "",
            "published": pd.Timestamp.now("UTC") - pd.Timedelta(days=2),
        }
    ]
    pack, _ = packs.build_pack("NVDA", "Official media", items, [])
    assert "(Reuters, 2d)" in pack


def test_news_items_nan_summary_and_publisher_safe() -> None:
    yf = pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "published": pd.to_datetime(["2026-06-10"], utc=True),
            "title": ["Bare headline"],
            "summary": [float("nan")],  # yfinance omits summaries; StringDtype stores None as NaN
            "publisher": [float("nan")],
        }
    )
    gn = pd.DataFrame(
        {
            "ticker": [],
            "published": pd.Series([], dtype="datetime64[ms, UTC]"),
            "title": [],
            "publisher": [],
            "url": [],
        }
    )
    items = packs.news_items(yf, gn)
    assert items[0]["text"] == "Bare headline"  # no " — nan"
    assert items[0]["source"] == "Yahoo Finance"


def test_viral_items_nan_sentiment_not_rendered() -> None:
    wsb = pd.DataFrame(
        {
            "ticker": [],
            "subreddit": [],
            "created": [],
            "title": [],
            "text": [],
            "score": [],
            "num_comments": [],
            "url": [],
        }
    )
    st = pd.DataFrame(
        {
            "ticker": ["NVDA"] * 2,
            "created": pd.to_datetime(["2026-06-10", "2026-06-09"], utc=True),
            "body": ["tagged msg", "untagged msg"],
            "sentiment": ["Bullish", float("nan")],  # NaN as if from a non-normalized frame
            "username": ["u1", "u2"],
            "url": ["https://st/1", "https://st/2"],
        }
    )
    items = packs.viral_items(wsb, st, _empty_bluesky())
    texts = [it["text"] for it in items]
    assert "tagged msg [user-tagged Bullish]" in texts
    assert "untagged msg" in texts  # no "[user-tagged nan]"


def _empty_bluesky() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [],
            "created": [],
            "text": [],
            "handle": [],
            "likes": [],
            "reposts": [],
            "url": [],
        }
    )


def test_viral_items_includes_bluesky_posts() -> None:
    empty_reddit = pd.DataFrame(
        {
            "ticker": [],
            "subreddit": [],
            "created": [],
            "title": [],
            "text": [],
            "score": [],
            "num_comments": [],
            "url": [],
        }
    )
    empty_st = pd.DataFrame(
        {"ticker": [], "created": [], "body": [], "sentiment": [], "username": [], "url": []}
    )
    bsky = pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "created": pd.to_datetime(["2026-06-10"], utc=True),
            "text": ["$NVDA to the sky 🚀"],
            "handle": ["bull.bsky.social"],
            "likes": [412],
            "reposts": [88],
            "url": ["https://bsky.app/profile/bull.bsky.social/post/1"],
        }
    )
    items = packs.viral_items(empty_reddit, empty_st, bsky)
    assert len(items) == 1
    assert items[0]["source"] == "Bluesky @bull.bsky.social (+412, 88r)"
    assert items[0]["text"] == "$NVDA to the sky 🚀"
    assert items[0]["url"] == "https://bsky.app/profile/bull.bsky.social/post/1"


def test_viral_items_round_robin_survives_pack_pressure() -> None:
    wsb = pd.DataFrame(
        {
            "ticker": ["NVDA"] * 20,
            "subreddit": ["wallstreetbets"] * 20,
            "created": pd.to_datetime(["2026-06-10"] * 20, utc=True),
            "title": [f"wsb long post {i} " + "x" * 300 for i in range(20)],
            "text": ["y" * 300] * 20,
            "score": [100] * 20,
            "num_comments": [50] * 20,
            "url": [f"https://reddit/{i}" for i in range(20)],
        }
    )
    st = pd.DataFrame(
        {
            "ticker": ["NVDA"] * 30,
            "created": pd.to_datetime(["2026-06-10"] * 30, utc=True),
            "body": [f"st long message {i} " + "z" * 300 for i in range(30)],
            "sentiment": ["Bullish"] * 30,
            "username": [f"u{i}" for i in range(30)],
            "url": [f"https://st/{i}" for i in range(30)],
        }
    )
    bsky = pd.DataFrame(
        {
            "ticker": ["NVDA"] * 3,
            "created": pd.to_datetime(["2026-06-10"] * 3, utc=True),
            "text": [f"bsky post {i} about $NVDA" for i in range(3)],
            "handle": [f"h{i}.bsky.social" for i in range(3)],
            "likes": [10] * 3,
            "reposts": [2] * 3,
            "url": [f"https://bsky.app/profile/h{i}/post/{i}" for i in range(3)],
        }
    )
    items = packs.viral_items(wsb, st, bsky)
    _, kept = packs.build_pack("NVDA", "Viral / retail", items, [])
    assert len(kept) < len(items)  # budget pressure is actually in play
    assert any(item["source"].startswith("Bluesky") for item in kept)
