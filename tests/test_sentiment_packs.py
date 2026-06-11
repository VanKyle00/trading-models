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
    items = packs.viral_items(wsb, st)
    texts = [it["text"] for it in items]
    assert "tagged msg [user-tagged Bullish]" in texts
    assert "untagged msg" in texts  # no "[user-tagged nan]"
