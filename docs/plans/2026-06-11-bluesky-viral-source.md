# Bluesky Viral-Tier Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Bluesky cashtag search as a fourth source in the ticker-sentiment tool's viral tier (keyless `api.bsky.app` searchPosts).

**Architecture:** One new loader following the house snapshot-cache convention, then a single wiring pass through the tier-3 seam the original design left open: `packs.viral_items` and `scoring.viral_metrics` gain a `bluesky` frame parameter, `report.py` gains one fetch job and the viral tier's source-status entry. No webapp changes — tier cards render new metrics/statuses generically.

**Tech Stack:** Python 3.12, pandas, httpx (already core deps — nothing new). Spec: `docs/specs/2026-06-11-bluesky-viral-source-design.md`.

---

## Repo primer

- Run everything with `uv run …` from the worktree root (`C:\Users\Administrator\trading-models\.claude\worktrees\bluesky-viral`). Tests: `uv run pytest <file> -q`. Lint/format: `uv run ruff check|format <paths>`.
- Loader convention: mirror `tradinglib/loaders/social/stocktwits.py` exactly (constants, typed `_empty()`, `_download` with catch-all → warn + empty, per-UTC-day parquet snapshot under `processed_dir(SOURCE)/_SUBDIR/<ticker>/`, `refresh` bypass, `head(max_items)`).
- House dtype rule: timestamp columns are `datetime64[ms, UTC]` in BOTH `_empty()` and built frames, via `.astype("datetime64[ms, UTC]")` with the comment `# ms (not ns) so the dtype survives the parquet round-trip and cached == fresh`.
- NaN rule: never truthiness-test pandas scalars; use `packs._str(...)` for possibly-NaN strings.
- Tests never hit the network: monkeypatch `processed_dir` on the loader module, stub `loader.httpx` via `SimpleNamespace`.
- All files UTF-8 (the fixture deliberately contains emoji).

## File map

```
tradinglib/loaders/social/bluesky.py     NEW   keyless searchPosts loader
tests/fixtures/sentiment/bluesky.json    NEW   recorded response w/ emoji + no-handle row
tests/test_bluesky_loader.py             NEW
tradinglib/sentiment/packs.py            MOD   viral_items gains bluesky param + item builder
tradinglib/sentiment/scoring.py          MOD   _TIER_DESC viral + viral_metrics gains bluesky
tradinglib/sentiment/report.py           MOD   import + fetch job + viral tier wiring
tests/test_sentiment_packs.py            MOD   signature update + bluesky item test
tests/test_sentiment_scoring.py          MOD   signature updates + bsky_mentions asserts
tests/test_sentiment_report.py           MOD   stub fixture + asserts + bluesky-error test
docs/data-sources.md                     MOD   Bluesky entry
README.md                                MOD   tier-3 row + viral-proxy sentence
```

---

### Task 1: Bluesky loader

**Files:**
- Create: `tradinglib/loaders/social/bluesky.py`
- Create: `tests/fixtures/sentiment/bluesky.json`
- Test: `tests/test_bluesky_loader.py`

- [ ] **Step 1: Write fixture + failing tests**

`tests/fixtures/sentiment/bluesky.json` (UTF-8; the emoji and the empty-author third post are deliberate):

```json
{
  "posts": [
    {
      "uri": "at://did:plc:abc123/app.bsky.feed.post/3kabc111",
      "author": {"handle": "chipbull.bsky.social"},
      "likeCount": 412,
      "repostCount": 88,
      "record": {
        "text": "$NVDA datacenter demand is unreal 🚀📈 loading more calls",
        "createdAt": "2026-06-10T14:30:00.000Z"
      }
    },
    {
      "uri": "at://did:plc:def456/app.bsky.feed.post/3kdef222",
      "author": {"handle": "quietbear.bsky.social"},
      "likeCount": 95,
      "repostCount": 12,
      "record": {
        "text": "Everyone euphoric on $NVDA again — fading this 🐻",
        "createdAt": "2026-06-09T11:05:00.000Z"
      }
    },
    {
      "uri": "at://did:plc:ghi789/app.bsky.feed.post/3kghi333",
      "author": {},
      "likeCount": 3,
      "repostCount": 0,
      "record": {
        "text": "$NVDA chart looking spicy",
        "createdAt": "2026-06-08T09:00:00.000Z"
      }
    }
  ]
}
```

`tests/test_bluesky_loader.py`:

```python
"""Tests for the Bluesky cashtag-search loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "sentiment" / "bluesky.json").read_text(
        encoding="utf-8"
    )
)


class _Resp:
    def __init__(self, payload: dict | None = None, fail: bool = False) -> None:
        self._payload = payload or {}
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 403")

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.social import bluesky as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_bluesky_schema_fields_and_params(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _get(url: str, **kwargs):
        seen["url"] = url
        seen["params"] = kwargs.get("params") or {}
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    df = loader.get_bluesky_posts("NVDA")
    assert list(df.columns) == ["ticker", "created", "text", "handle", "likes", "reposts", "url"]
    assert len(df) == 3
    assert df.iloc[0]["handle"] == "chipbull.bsky.social"  # API ("top") order preserved
    assert df.iloc[0]["likes"] == 412 and df.iloc[0]["reposts"] == 88
    assert df.iloc[0]["url"] == "https://bsky.app/profile/chipbull.bsky.social/post/3kabc111"
    assert "🚀" in df.iloc[0]["text"]  # UTF-8 survives end to end
    assert df.iloc[2]["url"] == ""  # no author handle -> no link
    assert str(df["created"].dt.tz) == "UTC"
    assert "app.bsky.feed.searchPosts" in seen["url"]
    assert seen["params"]["q"] == "$NVDA"
    assert seen["params"]["sort"] == "top"
    assert seen["params"]["lang"] == "en"
    assert "since" in seen["params"] and seen["params"]["limit"] == 25


def test_bluesky_http_error_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp(fail=True))
    )
    df = loader.get_bluesky_posts("NVDA")
    assert df.empty
    assert list(df.columns) == ["ticker", "created", "text", "handle", "likes", "reposts", "url"]


def test_bluesky_snapshot_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def _get(url: str, **kwargs):
        calls.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    first = loader.get_bluesky_posts("NVDA")
    second = loader.get_bluesky_posts("NVDA")
    assert len(calls) == 1
    assert first.equals(second)
    assert "🚀" in second.iloc[0]["text"]  # emoji survives the parquet round-trip


def test_bluesky_caps_items(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp(_FIXTURE)))
    df = loader.get_bluesky_posts("NVDA", max_items=1)
    assert len(df) == 1
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_bluesky_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradinglib.loaders.social.bluesky'`

- [ ] **Step 3: Implement the loader**

`tradinglib/loaders/social/bluesky.py`:

```python
"""Bluesky cashtag-search loader (Tier 3 viral source).

Schema (canonical): ``[ticker, created, text, handle, likes, reposts, url]``,
UTC-aware, kept in the API's engagement ("top") order — deliberately NOT
re-sorted by time, the ranking is the virality signal — capped at
``max_items``.

Keyless: ``api.bsky.app``'s searchPosts works unauthenticated (verified
2026-06-11; the deliberately-public host gates search, so this may close
someday). Any fetch failure logs and returns empty — tier 3 proceeds on
Stocktwits + Trends; an authenticated app-password session is the documented
upgrade path (see the design spec). Snapshot-cached to
``data/processed/social/bluesky/<ticker>/<snapshot>.parquet``.
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "social"
_SUBDIR = "bluesky"
_TIMEOUT_S = 8.0
_WINDOW_DAYS = 7
_UA = "Mozilla/5.0 (compatible; trading-models-sentiment/0.1)"

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "created": pd.Series([], dtype="datetime64[ms, UTC]"),
            "text": pd.Series([], dtype="object"),
            "handle": pd.Series([], dtype="object"),
            "likes": pd.Series([], dtype="int64"),
            "reposts": pd.Series([], dtype="int64"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _row(post: dict) -> dict:
    record = post.get("record") or {}
    handle = (post.get("author") or {}).get("handle", "")
    uri = post.get("uri", "")
    rkey = uri.rsplit("/", 1)[-1] if uri else ""
    return {
        "created": pd.to_datetime(record.get("createdAt"), utc=True, errors="coerce"),
        "text": record.get("text", ""),
        "handle": handle,
        "likes": int(post.get("likeCount") or 0),
        "reposts": int(post.get("repostCount") or 0),
        "url": f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else "",
    }


def _download(ticker: str, max_items: int) -> pd.DataFrame:
    since = (pd.Timestamp.now("UTC") - pd.Timedelta(days=_WINDOW_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    try:
        resp = httpx.get(
            "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={
                "q": f"${ticker}",
                "sort": "top",
                "since": since,
                "lang": "en",
                "limit": max_items,
            },
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        resp.raise_for_status()
        posts = resp.json().get("posts") or []
    except Exception:
        logger.warning("bluesky fetch failed for %s; returning empty", ticker, exc_info=True)
        posts = []
    rows = [_row(p) for p in posts if (p.get("record") or {}).get("text")]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    # ms (not ns) so the dtype survives the parquet round-trip and cached == fresh
    df["created"] = pd.to_datetime(df["created"], utc=True).astype("datetime64[ms, UTC]")
    return df  # keep the API's engagement order — the ranking IS the signal


def get_bluesky_posts(ticker: str, *, max_items: int = 25, refresh: bool = False) -> pd.DataFrame:
    """Top cashtag posts for one ticker over the last week, engagement-ranked."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker, max_items)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_bluesky_loader.py -q`
Expected: 4 passed

- [ ] **Step 5: Lint, format, commit**

Run `uv run ruff check tradinglib/loaders/social/bluesky.py tests/test_bluesky_loader.py` and `uv run ruff format` on the same files (re-run tests if format rewrites), then:

```bash
git add tradinglib/loaders/social/bluesky.py tests/fixtures/sentiment/bluesky.json tests/test_bluesky_loader.py
git commit -m "feat(sentiment): bluesky cashtag-search loader"
```

---

### Task 2: Tier-3 engine wiring

**Files:**
- Modify: `tradinglib/sentiment/packs.py` (viral_items)
- Modify: `tradinglib/sentiment/scoring.py` (_TIER_DESC, viral_metrics)
- Modify: `tradinglib/sentiment/report.py` (import, fetch job, viral tier def)
- Test: `tests/test_sentiment_packs.py`, `tests/test_sentiment_scoring.py`, `tests/test_sentiment_report.py`

This task is one commit: the signature changes ripple across the three modules, and intermediate states don't pass tests. Update tests first (Steps 1–2), then implement (Steps 3–5), then verify (Step 6).

- [ ] **Step 1: Update the three test files (failing first)**

(a) `tests/test_sentiment_packs.py` — in `test_viral_items_nan_sentiment_not_rendered`, change the call

```python
    items = packs.viral_items(wsb, st)
```

to

```python
    items = packs.viral_items(wsb, st, _empty_bluesky())
```

and add this helper plus a new test at the end of the file:

```python
def _empty_bluesky() -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": [], "created": [], "text": [], "handle": [], "likes": [],
         "reposts": [], "url": []}
    )


def test_viral_items_includes_bluesky_posts() -> None:
    empty_reddit = pd.DataFrame(
        {"ticker": [], "subreddit": [], "created": [], "title": [], "text": [],
         "score": [], "num_comments": [], "url": []}
    )
    empty_st = pd.DataFrame(
        {"ticker": [], "created": [], "body": [], "sentiment": [], "username": [],
         "url": []}
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
```

(b) `tests/test_sentiment_scoring.py` — in `test_viral_metrics_ratio_and_spike`, change

```python
    m = scoring.viral_metrics(wsb, st, interest)
```

to

```python
    bsky = pd.DataFrame({"text": ["a", "b", "c"]})
    m = scoring.viral_metrics(wsb, st, bsky, interest)
```

and add after the existing asserts in that test:

```python
    assert m["bsky_mentions"] == 3
```

In `test_viral_metrics_guards`, change

```python
    m = scoring.viral_metrics(pd.DataFrame(), st, None)
```

to

```python
    m = scoring.viral_metrics(pd.DataFrame(), st, pd.DataFrame(), None)
```

and add after its asserts:

```python
    assert m["bsky_mentions"] == 0
```

(c) `tests/test_sentiment_report.py` — add a frame builder after the `_st()` helper:

```python
def _bsky() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA"] * 2,
            "created": pd.to_datetime(["2026-06-10", "2026-06-09"], utc=True),
            "text": ["$NVDA to the sky 🚀", "fading $NVDA here"],
            "handle": ["bull.bsky.social", "bear.bsky.social"],
            "likes": [412, 95],
            "reposts": [88, 12],
            "url": [
                "https://bsky.app/profile/bull.bsky.social/post/1",
                "https://bsky.app/profile/bear.bsky.social/post/2",
            ],
        }
    )
```

In the `stubbed` fixture, add alongside the other loader patches:

```python
    monkeypatch.setattr(report_mod, "get_bluesky_posts", lambda t, **kw: _bsky())
```

In `test_full_report`, extend the viral assertions (after the `st_bull_bear_ratio` line):

```python
    assert by_tier["viral"].metrics["bsky_mentions"] == 2
    assert by_tier["viral"].source_status["bluesky"] == "ok"
```

In `test_all_sources_empty_is_no_data`, add alongside the other empty patches:

```python
    monkeypatch.setattr(report_mod, "get_bluesky_posts", lambda t, **kw: empty)
```

Add a new test after `test_source_error_degrades_tier`:

```python
def test_bluesky_error_degrades_viral(stubbed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(t, **kw):
        raise RuntimeError("bsky gated")

    monkeypatch.setattr(report_mod, "get_bluesky_posts", _boom)
    rep = report_mod.run_sentiment("NVDA", provider=_TierStub(_PROVIDER_PAYLOADS))
    viral = next(t for t in rep.tiers if t.tier == "viral")
    assert viral.status == "degraded"  # bluesky is pack content, unlike trends
    assert viral.source_status["bluesky"].startswith("error")
    assert viral.metrics["bsky_mentions"] == 0
    assert rep.status == "partial"
```

- [ ] **Step 2: Run the three suites, verify failures**

Run: `uv run pytest tests/test_sentiment_packs.py tests/test_sentiment_scoring.py tests/test_sentiment_report.py -q`
Expected: FAIL — `TypeError` on the new 3-/4-arg calls and `AttributeError: ... has no attribute 'get_bluesky_posts'`

- [ ] **Step 3: packs.py — viral_items gains the bluesky frame**

Replace the whole `viral_items` function with:

```python
def viral_items(
    wsb_posts: pd.DataFrame, stocktwits: pd.DataFrame, bluesky: pd.DataFrame
) -> list[dict]:
    """Tier-3 items: r/wallstreetbets posts + Stocktwits messages + Bluesky posts."""
    items: list[dict] = [_reddit_item(r) for r in wsb_posts.itertuples()]
    for r in stocktwits.itertuples():
        # NaN-safe: a non-normalized frame can carry float NaN, which is truthy
        tag = f" [user-tagged {_str(r.sentiment)}]" if _str(r.sentiment) else ""
        items.append(
            {
                "source": "Stocktwits",
                "title": str(r.body)[:80],
                "text": f"{r.body}{tag}",
                "url": r.url,
                "published": r.created,
            }
        )
    for r in bluesky.itertuples():
        text = _str(r.text)
        items.append(
            {
                "source": f"Bluesky @{_str(r.handle)} (+{int(r.likes)}, {int(r.reposts)}r)",
                "title": text[:80],
                "text": text,
                "url": r.url,
                "published": r.created,
            }
        )
    return _dedupe(items)
```

- [ ] **Step 4: scoring.py — tier description + viral_metrics**

In `_TIER_DESC`, change the `TIER_VIRAL` value to:

```python
    TIER_VIRAL: "viral retail chatter (r/wallstreetbets posts, Stocktwits messages, Bluesky posts)",
```

Replace the whole `viral_metrics` function with:

```python
def viral_metrics(
    wsb_posts: pd.DataFrame,
    stocktwits: pd.DataFrame,
    bluesky: pd.DataFrame,
    interest: pd.Series | None,
) -> dict:
    bulls = int(stocktwits["sentiment"].eq("Bullish").sum()) if len(stocktwits) else 0
    bears = int(stocktwits["sentiment"].eq("Bearish").sum()) if len(stocktwits) else 0
    return {
        "wsb_mentions": len(wsb_posts),
        "st_messages": len(stocktwits),
        "st_bullish": bulls,
        "st_bearish": bears,
        "st_bull_bear_ratio": round(bulls / bears, 2) if bears > 0 else None,
        "bsky_mentions": len(bluesky),
        "trends_spike": trends_spike(interest),
    }
```

- [ ] **Step 5: report.py — import, fetch job, viral tier def**

Add to the loaders import block (sorted — before the stocktwits import):

```python
from tradinglib.loaders.social.bluesky import get_bluesky_posts
```

In `_fetch_sources`'s `jobs` dict, add directly before the `"stocktwits"` entry:

```python
        "bluesky": lambda: get_bluesky_posts(ticker, max_items=25, refresh=refresh),
```

Replace the `TIER_VIRAL` tuple in `tiers_def` with:

```python
        (
            TIER_VIRAL,
            packs.viral_items(
                _frame(data, "wsb"), _frame(data, "stocktwits"), _frame(data, "bluesky")
            ),
            scoring.viral_metrics(
                _frame(data, "wsb"),
                _frame(data, "stocktwits"),
                _frame(data, "bluesky"),
                trends,
            ),
            {k: status[k] for k in ("wsb", "stocktwits", "bluesky", "google_trends")},
        ),
```

- [ ] **Step 6: Run the full sentiment + loader suites, verify pass**

Run: `uv run pytest tests/test_sentiment_packs.py tests/test_sentiment_scoring.py tests/test_sentiment_report.py tests/test_bluesky_loader.py tests/test_stocktwits_loader.py tests/test_reddit_loader.py -q`
Expected: all pass (packs 8, scoring 10, report 10, bluesky 4, stocktwits 3, reddit 4 → 39 passed)

- [ ] **Step 7: Lint, format, commit**

Run ruff check + format on the six changed files (re-run tests if format rewrites), then:

```bash
git add tradinglib/sentiment/packs.py tradinglib/sentiment/scoring.py tradinglib/sentiment/report.py tests/test_sentiment_packs.py tests/test_sentiment_scoring.py tests/test_sentiment_report.py
git commit -m "feat(sentiment): wire bluesky into the viral tier"
```

---

### Task 3: Docs

**Files:**
- Modify: `docs/data-sources.md` (new entry after the Stocktwits entry, still inside "Currently wired in")
- Modify: `README.md` (tier-3 table row + the viral-proxy sentence in the sentiment section)

- [ ] **Step 1: data-sources.md entry**

Append directly after the Stocktwits entry (before "## Planned / not yet wired in"):

```markdown
### Bluesky — cashtag post search

- **What**: Top posts mentioning `$<TICKER>` over the last week
  (`[ticker, created, text, handle, likes, reposts, url]`), engagement-ranked.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.social.bluesky`](../tradinglib/loaders/social/bluesky.py)
- **Notes**: Keyless search on `api.bsky.app` (verified 2026-06-11; the
  deliberately-public AppView host gates search, so this endpoint may close
  someday — failures degrade the viral tier gracefully, and an authenticated
  app-password session is the documented upgrade path). Tier-3 source for the
  `/sentiment` page alongside r/wallstreetbets and Stocktwits.
```

- [ ] **Step 2: README updates**

In the sentiment section's tier table, replace the tier-3 row

```markdown
| **3 · Viral retail** | r/wallstreetbets + Stocktwits (user-tagged bull/bear) + Google Trends | bull/bear ratio, WSB mentions, search-spike ratio (7d vs ~90d) |
```

with

```markdown
| **3 · Viral retail** | r/wallstreetbets + Stocktwits (user-tagged bull/bear) + Bluesky cashtag search + Google Trends | bull/bear ratio, WSB + Bluesky mentions, search-spike ratio (7d vs ~90d) |
```

In the "Free sources only, honest degradation" bullet, change

```markdown
  Stocktwits + Trends.
```

(the end of the sentence "the viral tier is proxied by WSB + Stocktwits + Trends.") to

```markdown
  Stocktwits + Bluesky + Trends.
```

- [ ] **Step 3: Commit**

```bash
git add docs/data-sources.md README.md
git commit -m "docs: bluesky viral-tier source"
```

---

### Task 4: Full verification gate

- [ ] **Step 1: Run the CI gate locally**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy tradinglib
uv run pytest -q
```

Expected: all clean; pytest total = previous main count + 6 new tests (4 loader + 1 packs + 1 report; net of signature-only edits). Then the two CI extras exactly as `.github/workflows/ci.yml` defines them (Streamlit import check, MODELS.md freshness) — both untouched by this change.

- [ ] **Step 2 (optional live smoke, network + ANTHROPIC_API_KEY required):**

```bash
uv run python -c "from tradinglib.sentiment.report import run_sentiment; r = run_sentiment('NVDA', refresh=True); v = [t for t in r.tiers if t.tier == 'viral'][0]; print(v.status, v.metrics.get('bsky_mentions'), v.source_status)"
```

Expected: viral tier with a nonzero `bsky_mentions` and `bluesky: ok` in source_status.

- [ ] **Step 3: Commit any gate fixes**

```bash
git add -A && git commit -m "chore: verification gate fixes"  # only if the gate changed files
```

---

## Plan self-review notes (already applied)

- Spec coverage: loader contract incl. no-resort decision + ms dtype + NaN safety (Task 1), engine wiring with bluesky-degrades-unlike-trends semantics pinned by a test (Task 2), UTF-8/emoji fixture (Task 1), docs + README rows (Task 3), keyless caveat documented (Task 3). Non-goals untouched (no auth path, no webapp changes).
- Signature consistency: `viral_items(wsb, stocktwits, bluesky)` and `viral_metrics(wsb, stocktwits, bluesky, interest)` are used identically in Tasks 1–2 code and all test updates.
- The report-test stub `_bsky()` frames carry full loader schema so `itertuples` attribute access (`r.likes`, `r.handle`) works in packs.
- `test_full_report`'s existing `len(provider.calls) == 3` and divergence asserts are unchanged — bluesky adds items to the viral pack but doesn't change tier count or stub routing.
