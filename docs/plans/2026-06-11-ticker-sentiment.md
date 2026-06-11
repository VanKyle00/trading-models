# Ticker Sentiment (Three-Tier Read) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On-demand `/sentiment` webapp page: type a ticker, get a three-tier sentiment read (official media / serious forums / viral retail) — each tier scored by one bounded LLM call plus mechanical metrics, with a deterministic aggregate and a tier-divergence callout.

**Architecture:** Four new free-source loaders (Google News RSS, Seeking Alpha RSS, Reddit via praw, Stocktwits) plus two existing loaders (yfinance news, Google Trends) feed bounded per-tier text packs. `tradinglib/sentiment/` scores each non-empty tier with one strict-JSON provider call (the `scanner/briefs.py` pattern), resolves LLM evidence *indices* back to real rows (no hallucinated URLs), aggregates deterministically, and caches the finished report per (ticker, day). The webapp adds one page + one JSON API route.

**Tech Stack:** Python 3.12, pandas, httpx + feedparser (RSS), praw (Reddit), FastAPI + Jinja2, Anthropic provider behind the repo's `LLMProvider` protocol. Spec: `docs/specs/2026-06-11-ticker-sentiment-design.md`.

---

## Repo primer (read once before Task 1)

- Run everything with `uv run …` from the repo root. Tests: `uv run pytest -q`. The dev extra must be installed: `uv sync --extra dev`.
- **Loader convention** (see `tradinglib/loaders/news/yfinance.py`): module-level `SOURCE`/`_SUBDIR`, a `_empty()` returning a typed empty DataFrame, a `_download(...)` that catches all exceptions and logs+returns empty, and a public `get_x(ticker, *, max_items, refresh)` that snapshot-caches per UTC day to `processed_dir(SOURCE)/_SUBDIR/<ticker>/<YYYY-MM-DD>.parquet`.
- **Tests never hit the network.** Loader tests `monkeypatch.setattr(loader_module, "processed_dir", lambda source: tmp_path / source)` and stub the third-party boundary (`loader.httpx`, `loader._make_reddit`, `loader.yf.Ticker`).
- **LLM calls** go through `tradinglib.assistant.provider.LLMProvider` protocol: `provider.complete(system: str, [UserMsg(text)], tools=[]) -> AssistantTurn` where `turn.text` is the reply. `ClaudeProvider()` is the real one (lazy anthropic import; honors `ASSISTANT_MODEL`, default `claude-haiku-4-5`). `StubProvider([AssistantTurn(...)])` is for tests. `AssistantTurn(text=..., tool_calls=(), stop_reason="end_turn", usage=Usage(100, 100))`.
- **Webapp**: routes live inside `create_app()` in `webapp/main.py`; templates are standalone HTML files in `webapp/templates/` with inline CSS (copy the theme scaffold from `scans.html`); view helpers live in `webapp/<page>.py`. Tests use `TestClient(create_app())`.
- ruff: line-length 100, rules `E,F,I,N,UP,B,SIM,RUF`. mypy is non-strict and only checks `tradinglib/`. `ignore_missing_imports = true` (feedparser/praw need no stubs).

## File map

```
pyproject.toml                                    MOD   feedparser+praw core; drop ingest-extra
tradinglib/loaders/news/google_news.py            NEW   Tier-1: Google News RSS
tradinglib/loaders/forums/__init__.py             NEW
tradinglib/loaders/forums/seeking_alpha.py        NEW   Tier-2: SA per-ticker RSS (titles)
tradinglib/loaders/forums/reddit.py               NEW   Tier-2/3: Reddit search (praw)
tradinglib/loaders/social/__init__.py             NEW
tradinglib/loaders/social/stocktwits.py           NEW   Tier-3: ST symbol stream
tradinglib/sentiment/__init__.py                  NEW
tradinglib/sentiment/types.py                     NEW   Evidence/TierReport/SentimentReport
tradinglib/sentiment/packs.py                     NEW   items builders + bounded pack
tradinglib/sentiment/scoring.py                   NEW   score_tier + mechanical metrics
tradinglib/sentiment/report.py                    NEW   run_sentiment orchestrator
webapp/sentiment.py                               NEW   ticker validation + engine call
webapp/templates/sentiment.html                   NEW   page
webapp/main.py                                    MOD   2 routes + import
webapp/templates/{scans,planner,tournaments,tournament_day,models,index}.html  MOD  nav crumb
docs/data-sources.md                              MOD   4 new source entries
docs/DEPLOY.md                                    MOD   REDDIT_* env vars
tests/fixtures/sentiment/{google_news.xml,seeking_alpha.xml,stocktwits.json}   NEW
tests/test_{google_news,seeking_alpha,stocktwits,reddit}_loader.py            NEW
tests/test_sentiment_{types,packs,scoring,report}.py                          NEW
tests/test_webapp_sentiment.py                                                NEW
```

---

### Task 1: Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move praw to core deps, add feedparser**

In `pyproject.toml`, append to the `dependencies = [...]` list (after the `"anthropic>=0.69",` line):

```toml
    # Sentiment tool sources (RSS feeds + Reddit) — free/keyless except Reddit OAuth
    "feedparser>=6.0",
    "praw>=7.7",
```

Then delete the now-redundant `ingest-extra` block entirely (praw's single declaration moves to core):

```toml
ingest-extra = [
    "praw>=7.7",  # historical Reddit posts; current models don't import praw
]
```

(`grep -rn "ingest-extra" --include="*.md" .` — if any doc references it, update that sentence to say praw is now a core dependency. As of plan-writing only the spec mentions it, and the spec already says "moves to core".)

- [ ] **Step 2: Re-lock and sync**

Run: `uv lock` then `uv sync --extra dev`
Expected: lockfile updates; `feedparser` and `praw` install.

- [ ] **Step 3: Verify imports**

Run: `uv run python -c "import feedparser, praw; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): feedparser + praw as core deps for sentiment sources"
```

---

### Task 2: Google News RSS loader (Tier 1)

**Files:**
- Create: `tradinglib/loaders/news/google_news.py`
- Create: `tests/fixtures/sentiment/google_news.xml`
- Test: `tests/test_google_news_loader.py`

- [ ] **Step 1: Write fixture + failing tests**

`tests/fixtures/sentiment/google_news.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>"NVDA stock" - Google News</title>
    <item>
      <title>Nvidia rallies on record data-center revenue</title>
      <link>https://news.example.com/nvda-rally</link>
      <pubDate>Wed, 10 Jun 2026 14:00:00 GMT</pubDate>
      <source url="https://reuters.com">Reuters</source>
    </item>
    <item>
      <title>Analysts split on Nvidia valuation after run-up</title>
      <link>https://news.example.com/nvda-split</link>
      <pubDate>Tue, 09 Jun 2026 09:30:00 GMT</pubDate>
      <source url="https://cnbc.com">CNBC</source>
    </item>
  </channel>
</rss>
```

`tests/test_google_news_loader.py`:

```python
"""Tests for the Google News RSS loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = (Path(__file__).parent / "fixtures" / "sentiment" / "google_news.xml").read_text(
    encoding="utf-8"
)


class _Resp:
    def __init__(self, text: str = "", fail: bool = False) -> None:
        self.text = text
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 503")


def _fake_httpx(text: str = _FIXTURE, *, fail: bool = False, calls: list | None = None):
    def _get(url: str, **kwargs):
        if calls is not None:
            calls.append(url)
        return _Resp(text, fail=fail)

    return SimpleNamespace(get=_get)


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.news import google_news as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_google_news_schema_and_order(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(loader, "httpx", _fake_httpx(calls=calls))
    df = loader.get_google_news("NVDA")
    assert list(df.columns) == ["ticker", "published", "title", "publisher", "url"]
    assert len(df) == 2
    assert df.iloc[0]["publisher"] == "Reuters"  # newest first
    assert df.iloc[0]["url"] == "https://news.example.com/nvda-rally"
    assert str(df["published"].dt.tz) == "UTC"
    assert "rss/search" in calls[0] and "NVDA+stock" in calls[0]


def test_google_news_http_error_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", _fake_httpx(fail=True))
    df = loader.get_google_news("NVDA")
    assert df.empty
    assert list(df.columns) == ["ticker", "published", "title", "publisher", "url"]


def test_google_news_snapshot_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(loader, "httpx", _fake_httpx(calls=calls))
    first = loader.get_google_news("NVDA")
    second = loader.get_google_news("NVDA")
    assert len(calls) == 1
    assert first.equals(second)


def test_google_news_caps_items(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "httpx", _fake_httpx())
    df = loader.get_google_news("NVDA", max_items=1)
    assert len(df) == 1
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_google_news_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradinglib.loaders.news.google_news'`

- [ ] **Step 3: Implement the loader**

`tradinglib/loaders/news/google_news.py`:

```python
"""Google News RSS loader — Tier-1 headline search for one ticker.

Schema (canonical): ``[ticker, published, title, publisher, url]`` with
``published`` UTC-aware, newest first, capped at ``max_items``. Query is
``<TICKER> stock when:14d`` — the "stock" suffix disambiguates single-letter
tickers (people search "F stock" too) and ``when:`` bounds the window.
Keyless public feed. Snapshot-cached to
``data/processed/news/google_news/<ticker>/<snapshot>.parquet``.
httpx is stubbed in tests (repo convention: no live network).
"""

from __future__ import annotations

import calendar
import logging
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "news"
_SUBDIR = "google_news"
_TIMEOUT_S = 8.0
_WINDOW_DAYS = 14

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "published": pd.Series([], dtype="datetime64[ns, UTC]"),
            "title": pd.Series([], dtype="object"),
            "publisher": pd.Series([], dtype="object"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _entry_published(entry: Any) -> Any:
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        return pd.NaT
    return pd.Timestamp(calendar.timegm(parsed), unit="s", tz="UTC")


def _download(ticker: str) -> pd.DataFrame:
    query = quote_plus(f"{ticker} stock when:{_WINDOW_DAYS}d")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT_S, follow_redirects=True)
        resp.raise_for_status()
        entries = feedparser.parse(resp.text).entries or []
    except Exception:
        logger.warning("google news fetch failed for %s; returning empty", ticker, exc_info=True)
        entries = []
    rows = [
        {
            "published": _entry_published(e),
            "title": getattr(e, "title", ""),
            "publisher": getattr(getattr(e, "source", None), "title", "") or "Google News",
            "url": getattr(e, "link", ""),
        }
        for e in entries
        if getattr(e, "title", "")
    ]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df.sort_values("published", ascending=False).reset_index(drop=True)


def get_google_news(ticker: str, *, max_items: int = 25, refresh: bool = False) -> pd.DataFrame:
    """Recent Google News headlines for one ticker, newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_google_news_loader.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tradinglib/loaders/news/google_news.py tests/fixtures/sentiment/google_news.xml tests/test_google_news_loader.py
git commit -m "feat(sentiment): google news rss loader"
```

---

### Task 3: Seeking Alpha RSS loader (Tier 2)

**Files:**
- Create: `tradinglib/loaders/forums/__init__.py`, `tradinglib/loaders/forums/seeking_alpha.py`
- Create: `tests/fixtures/sentiment/seeking_alpha.xml`
- Test: `tests/test_seeking_alpha_loader.py`

- [ ] **Step 1: Write fixture + failing tests**

`tests/fixtures/sentiment/seeking_alpha.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>NVDA on Seeking Alpha</title>
    <item>
      <title>Nvidia: The Data Center Story Is Not Priced In</title>
      <link>https://seekingalpha.com/article/0001-nvda-dc</link>
      <pubDate>Wed, 10 Jun 2026 11:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Nvidia Q1 Earnings: Time To Take Profits</title>
      <link>https://seekingalpha.com/article/0002-nvda-tp</link>
      <pubDate>Mon, 08 Jun 2026 16:20:00 GMT</pubDate>
    </item>
  </channel>
</rss>
```

`tests/test_seeking_alpha_loader.py`:

```python
"""Tests for the Seeking Alpha RSS loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = (Path(__file__).parent / "fixtures" / "sentiment" / "seeking_alpha.xml").read_text(
    encoding="utf-8"
)


class _Resp:
    def __init__(self, text: str = "", fail: bool = False) -> None:
        self.text = text
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 403")


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.forums import seeking_alpha as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_seeking_alpha_schema(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _get(url: str, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers") or {}
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    df = loader.get_seeking_alpha("NVDA")
    assert list(df.columns) == ["ticker", "published", "title", "url"]
    assert len(df) == 2
    assert df.iloc[0]["title"].startswith("Nvidia: The Data Center")
    assert "combined/NVDA.xml" in seen["url"]
    assert "User-Agent" in seen["headers"]


def test_seeking_alpha_blocked_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp("", fail=True))
    )
    df = loader.get_seeking_alpha("NVDA")
    assert df.empty
    assert list(df.columns) == ["ticker", "published", "title", "url"]


def test_seeking_alpha_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def _get(url: str, **kwargs):
        calls.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    loader.get_seeking_alpha("NVDA")
    loader.get_seeking_alpha("NVDA")
    assert len(calls) == 1
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_seeking_alpha_loader.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`tradinglib/loaders/forums/__init__.py`:

```python
"""Forum-post loaders."""
```

`tradinglib/loaders/forums/seeking_alpha.py`:

```python
"""Seeking Alpha per-ticker RSS loader — article titles only (Tier 2).

Schema (canonical): ``[ticker, published, title, url]``, UTC-aware, newest
first, capped at ``max_items``. Seeking Alpha has no public API; this is its
public per-ticker RSS feed (titles + links, no bodies). It is the most fragile
source in the sentiment tool — Cloudflare moods — so any failure logs and
returns empty (tier 2 then runs on Reddit alone). Snapshot-cached to
``data/processed/forums/seeking_alpha/<ticker>/<snapshot>.parquet``.
"""

from __future__ import annotations

import calendar
import logging
from typing import Any

import feedparser
import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "forums"
_SUBDIR = "seeking_alpha"
_TIMEOUT_S = 8.0
_UA = "Mozilla/5.0 (compatible; trading-models-sentiment/0.1)"

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "published": pd.Series([], dtype="datetime64[ns, UTC]"),
            "title": pd.Series([], dtype="object"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _entry_published(entry: Any) -> Any:
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        return pd.NaT
    return pd.Timestamp(calendar.timegm(parsed), unit="s", tz="UTC")


def _download(ticker: str) -> pd.DataFrame:
    url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
    try:
        resp = httpx.get(
            url, timeout=_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": _UA}
        )
        resp.raise_for_status()
        entries = feedparser.parse(resp.text).entries or []
    except Exception:
        logger.warning("seeking alpha fetch failed for %s; returning empty", ticker, exc_info=True)
        entries = []
    rows = [
        {
            "published": _entry_published(e),
            "title": getattr(e, "title", ""),
            "url": getattr(e, "link", ""),
        }
        for e in entries
        if getattr(e, "title", "")
    ]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df.sort_values("published", ascending=False).reset_index(drop=True)


def get_seeking_alpha(ticker: str, *, max_items: int = 20, refresh: bool = False) -> pd.DataFrame:
    """Recent Seeking Alpha article titles for one ticker, newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_seeking_alpha_loader.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tradinglib/loaders/forums/ tests/fixtures/sentiment/seeking_alpha.xml tests/test_seeking_alpha_loader.py
git commit -m "feat(sentiment): seeking alpha rss loader"
```

---

### Task 4: Stocktwits loader (Tier 3)

**Files:**
- Create: `tradinglib/loaders/social/__init__.py`, `tradinglib/loaders/social/stocktwits.py`
- Create: `tests/fixtures/sentiment/stocktwits.json`
- Test: `tests/test_stocktwits_loader.py`

- [ ] **Step 1: Write fixture + failing tests**

`tests/fixtures/sentiment/stocktwits.json`:

```json
{
  "response": {"status": 200},
  "symbol": {"id": 686, "symbol": "NVDA"},
  "messages": [
    {
      "id": 101,
      "body": "Loading calls into earnings, this thing rips",
      "created_at": "2026-06-10T14:01:00Z",
      "user": {"username": "bull_guy"},
      "entities": {"sentiment": {"basic": "Bullish"}}
    },
    {
      "id": 102,
      "body": "Way overextended here, fading the pump",
      "created_at": "2026-06-10T13:55:00Z",
      "user": {"username": "fade_king"},
      "entities": {"sentiment": {"basic": "Bearish"}}
    },
    {
      "id": 103,
      "body": "Anyone watching the morning level?",
      "created_at": "2026-06-10T13:50:00Z",
      "user": {"username": "lurker"},
      "entities": {"sentiment": null}
    }
  ]
}
```

`tests/test_stocktwits_loader.py`:

```python
"""Tests for the Stocktwits symbol-stream loader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "sentiment" / "stocktwits.json").read_text(
        encoding="utf-8"
    )
)


class _Resp:
    def __init__(self, payload: dict | None = None, fail: bool = False) -> None:
        self._payload = payload or {}
        self._fail = fail

    def raise_for_status(self) -> None:
        if self._fail:
            raise RuntimeError("http 404")

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tradinglib.loaders.social import stocktwits as mod

    monkeypatch.setattr(mod, "processed_dir", lambda source: tmp_path / source)
    return mod


def test_stocktwits_schema_and_tags(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    def _get(url: str, **kwargs):
        seen.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    df = loader.get_stocktwits("NVDA")
    assert list(df.columns) == ["ticker", "created", "body", "sentiment", "username", "url"]
    assert len(df) == 3
    assert list(df["sentiment"]) == ["Bullish", "Bearish", None]
    assert df.iloc[0]["url"] == "https://stocktwits.com/bull_guy/message/101"
    assert str(df["created"].dt.tz) == "UTC"
    assert "streams/symbol/NVDA.json" in seen[0]


def test_stocktwits_unknown_symbol_is_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loader, "httpx", SimpleNamespace(get=lambda url, **kw: _Resp(fail=True))
    )
    df = loader.get_stocktwits("ZZZZZZ")
    assert df.empty
    assert list(df.columns) == ["ticker", "created", "body", "sentiment", "username", "url"]


def test_stocktwits_cached(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def _get(url: str, **kwargs):
        calls.append(url)
        return _Resp(_FIXTURE)

    monkeypatch.setattr(loader, "httpx", SimpleNamespace(get=_get))
    loader.get_stocktwits("NVDA")
    loader.get_stocktwits("NVDA")
    assert len(calls) == 1
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_stocktwits_loader.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`tradinglib/loaders/social/__init__.py`:

```python
"""Social / retail message-stream loaders."""
```

`tradinglib/loaders/social/stocktwits.py`:

```python
"""Stocktwits symbol-stream loader (Tier 3 viral proxy).

Schema (canonical): ``[ticker, created, body, sentiment, username, url]``,
UTC-aware, newest first, capped at ``max_items``. ``sentiment`` is the
user-tagged label ("Bullish"/"Bearish") or ``None`` — free ground truth the
mechanical bull/bear ratio is computed from. Keyless public endpoint
(~200 req/hr/IP). Unknown symbols 404 → empty. Snapshot-cached to
``data/processed/social/stocktwits/<ticker>/<snapshot>.parquet``.
"""

from __future__ import annotations

import logging

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "social"
_SUBDIR = "stocktwits"
_TIMEOUT_S = 8.0

logger = logging.getLogger(__name__)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "created": pd.Series([], dtype="datetime64[ns, UTC]"),
            "body": pd.Series([], dtype="object"),
            "sentiment": pd.Series([], dtype="object"),
            "username": pd.Series([], dtype="object"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _row(message: dict) -> dict:
    sentiment = ((message.get("entities") or {}).get("sentiment") or {}).get("basic")
    username = (message.get("user") or {}).get("username", "")
    msg_id = message.get("id", "")
    return {
        "created": pd.to_datetime(message.get("created_at"), utc=True, errors="coerce"),
        "body": message.get("body", ""),
        "sentiment": sentiment,
        "username": username,
        "url": f"https://stocktwits.com/{username}/message/{msg_id}" if username else "",
    }


def _download(ticker: str) -> pd.DataFrame:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT_S, follow_redirects=True)
        resp.raise_for_status()
        messages = resp.json().get("messages") or []
    except Exception:
        logger.warning("stocktwits fetch failed for %s; returning empty", ticker, exc_info=True)
        messages = []
    rows = [_row(m) for m in messages if m.get("body")]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    df["created"] = pd.to_datetime(df["created"], utc=True)
    return df.sort_values("created", ascending=False).reset_index(drop=True)


def get_stocktwits(ticker: str, *, max_items: int = 30, refresh: bool = False) -> pd.DataFrame:
    """Recent Stocktwits messages for one ticker, newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_stocktwits_loader.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tradinglib/loaders/social/ tests/fixtures/sentiment/stocktwits.json tests/test_stocktwits_loader.py
git commit -m "feat(sentiment): stocktwits stream loader"
```

---

### Task 5: Reddit loader (Tiers 2+3)

**Files:**
- Create: `tradinglib/loaders/forums/reddit.py`
- Test: `tests/test_reddit_loader.py`

- [ ] **Step 1: Write failing tests**

`tests/test_reddit_loader.py`:

```python
"""Tests for the Reddit search loader (praw stubbed at the client boundary)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _submission(title: str, *, score: int = 100, comments: int = 20, selftext: str = "") -> SimpleNamespace:
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
        "ticker", "subreddit", "created", "title", "text", "score", "num_comments", "url",
    ]
    assert len(df) == 2
    assert set(df["subreddit"]) == {"stocks", "investing"}
    assert df.iloc[0]["url"].startswith("https://www.reddit.com/r/")
    assert str(df["created"].dt.tz) == "UTC"
    sub, query, sort, time_filter, limit = fake.queries[0]
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
        "ticker", "subreddit", "created", "title", "text", "score", "num_comments", "url",
    ]
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_reddit_loader.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`tradinglib/loaders/forums/reddit.py`:

```python
"""Reddit search loader — posts mentioning a ticker in given subreddits.

Serves both sentiment tiers: the engine passes serious-investing subreddits
for Tier 2 and r/wallstreetbets for Tier 3 (the tier->subreddit mapping lives
in ``tradinglib/sentiment/report.py``, not here).

Schema (canonical): ``[ticker, subreddit, created, title, text, score,
num_comments, url]``, UTC-aware, newest first. ``text`` is the selftext capped
at 500 chars. Cached per subreddit to
``data/processed/forums/reddit/<subreddit>/<ticker>/<snapshot>.parquet`` —
cache hits need no credentials.

Credentials: free OAuth app (script type) from reddit.com/prefs/apps via
``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` (+ optional
``REDDIT_USER_AGENT``). Missing credentials raise
``MissingRedditCredentials`` on uncached fetches; the sentiment engine
catches it and degrades the tier rather than crashing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "forums"
_SUBDIR = "reddit"
_TEXT_MAX_CHARS = 500

logger = logging.getLogger(__name__)


class MissingRedditCredentials(RuntimeError):
    """REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not configured."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "subreddit": pd.Series([], dtype="object"),
            "created": pd.Series([], dtype="datetime64[ns, UTC]"),
            "title": pd.Series([], dtype="object"),
            "text": pd.Series([], dtype="object"),
            "score": pd.Series([], dtype="int64"),
            "num_comments": pd.Series([], dtype="int64"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _make_reddit() -> Any:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise MissingRedditCredentials(
            "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set "
            "(create a script app at reddit.com/prefs/apps)"
        )
    import praw  # lazy: only needed for uncached fetches with credentials

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=os.environ.get("REDDIT_USER_AGENT", "trading-models-sentiment/0.1"),
    )


def _download_sub(reddit: Any, ticker: str, sub: str, limit: int) -> pd.DataFrame:
    try:
        submissions = list(
            reddit.subreddit(sub).search(
                f"{ticker} OR ${ticker}", sort="relevance", time_filter="week", limit=limit
            )
        )
    except Exception:
        logger.warning("reddit search failed for %s in r/%s; empty", ticker, sub, exc_info=True)
        submissions = []
    rows = [
        {
            "subreddit": sub,
            "created": pd.Timestamp(s.created_utc, unit="s", tz="UTC"),
            "title": s.title,
            "text": (s.selftext or "")[:_TEXT_MAX_CHARS],
            "score": int(s.score),
            "num_comments": int(s.num_comments),
            "url": f"https://www.reddit.com{s.permalink}",
        }
        for s in submissions
    ]
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df.insert(0, "ticker", ticker)
    return df


def get_reddit_posts(
    ticker: str, subreddits: tuple[str, ...], *, limit: int = 20, refresh: bool = False
) -> pd.DataFrame:
    """Posts mentioning ``ticker`` across ``subreddits`` (last week), newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    reddit: Any = None
    for sub in subreddits:
        out = processed_dir(SOURCE) / _SUBDIR / sub / ticker / f"{snapshot}.parquet"
        if out.exists() and not refresh:
            frames.append(pd.read_parquet(out))
            continue
        if reddit is None:
            reddit = _make_reddit()  # raises MissingRedditCredentials when unconfigured
        df = _download_sub(reddit, ticker, sub, limit)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else _empty()
    if combined.empty:
        return _empty()
    return combined.sort_values("created", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_reddit_loader.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tradinglib/loaders/forums/reddit.py tests/test_reddit_loader.py
git commit -m "feat(sentiment): reddit posts loader"
```

---

### Task 6: Report types

**Files:**
- Create: `tradinglib/sentiment/__init__.py`, `tradinglib/sentiment/types.py`
- Test: `tests/test_sentiment_types.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sentiment_types.py`:

```python
"""Round-trip tests for the sentiment report dataclasses."""

from __future__ import annotations

from tradinglib.sentiment.types import Evidence, SentimentReport, TierReport


def _report() -> SentimentReport:
    return SentimentReport(
        ticker="NVDA",
        as_of="2026-06-11T15:00:00+00:00",
        status="partial",
        tiers=[
            TierReport(
                tier="official",
                label="Official media",
                status="ok",
                score=0.6,
                stance="bullish",
                confidence=0.8,
                summary="Coverage is upbeat.",
                key_themes=["data center demand"],
                evidence=[Evidence(title="Nvidia rallies", source="Reuters", url="https://x", age_days=1.0)],
                metrics={"headline_count": 12},
                source_status={"yfinance_news": "ok", "google_news": "ok"},
                item_count=12,
            ),
            TierReport(tier="forums", label="Serious forums", status="no_data"),
            TierReport(tier="viral", label="Viral / retail", status="degraded", parse_error=True),
        ],
        overall_bias=0.6,
        divergence=None,
    )


def test_roundtrip() -> None:
    report = _report()
    raw = report.to_dict()
    back = SentimentReport.from_dict(raw)
    assert back == report
    assert raw["tiers"][0]["evidence"][0]["source"] == "Reuters"


def test_defaults() -> None:
    tier = TierReport(tier="forums", label="Serious forums", status="no_data")
    assert tier.score is None and tier.evidence == [] and tier.item_count == 0
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_sentiment_types.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`tradinglib/sentiment/__init__.py` (re-exports are added in Task 9; keep it minimal now):

```python
"""Three-tier ticker sentiment engine (official / forums / viral)."""
```

`tradinglib/sentiment/types.py`:

```python
"""Typed report objects for the three-tier sentiment read.

``SentimentReport.to_dict()`` must stay JSON-serializable — it is written to
``data/processed/sentiment/reports/<ticker>/<date>.json`` and served verbatim
by the webapp API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TIER_OFFICIAL = "official"
TIER_FORUMS = "forums"
TIER_VIRAL = "viral"
TIER_LABELS = {
    TIER_OFFICIAL: "Official media",
    TIER_FORUMS: "Serious forums",
    TIER_VIRAL: "Viral / retail",
}

# Two tiers whose scores differ by at least this much trigger the divergence callout.
DIVERGENCE_GAP = 0.6


@dataclass
class Evidence:
    title: str
    source: str
    url: str
    age_days: float | None


@dataclass
class TierReport:
    tier: str  # TIER_OFFICIAL | TIER_FORUMS | TIER_VIRAL
    label: str
    status: str  # "ok" | "degraded" | "no_data"
    score: float | None = None  # -1..1
    stance: str | None = None  # "bearish" | "neutral" | "bullish" | "mixed"
    confidence: float | None = None  # 0..1
    summary: str = ""
    key_themes: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    source_status: dict[str, str] = field(default_factory=dict)  # source -> ok|empty|error: …
    item_count: int = 0
    parse_error: bool = False


@dataclass
class SentimentReport:
    ticker: str
    as_of: str  # ISO-8601 UTC
    status: str  # "ok" | "partial" | "no_data"
    tiers: list[TierReport]
    overall_bias: float | None = None  # mean of available tier scores
    divergence: dict[str, Any] | None = None  # {"pair": [hi_tier, lo_tier], "gap": float}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SentimentReport:
        tiers = [
            TierReport(
                **{**t, "evidence": [Evidence(**e) for e in t.get("evidence") or []]}
            )
            for t in raw.get("tiers") or []
        ]
        return cls(
            ticker=raw["ticker"],
            as_of=raw["as_of"],
            status=raw["status"],
            tiers=tiers,
            overall_bias=raw.get("overall_bias"),
            divergence=raw.get("divergence"),
        )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_sentiment_types.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tradinglib/sentiment/ tests/test_sentiment_types.py
git commit -m "feat(sentiment): report types"
```

---

### Task 7: Pack assembly

**Files:**
- Create: `tradinglib/sentiment/packs.py`
- Test: `tests/test_sentiment_packs.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sentiment_packs.py`:

```python
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
        [{"ticker": "NVDA", "published": "2026-06-10", "title": "Nvidia rallies on revenue!",
          "summary": "Big quarter.", "publisher": "Reuters"}]
    )
    gn = _news_df(
        [{"ticker": "NVDA", "published": "2026-06-10", "title": "NVIDIA RALLIES ON REVENUE",
          "publisher": "Reuters", "url": "https://x"},
         {"ticker": "NVDA", "published": "2026-06-09", "title": "A different story",
          "publisher": "CNBC", "url": "https://y"}]
    )
    items = packs.news_items(yf, gn)
    assert len(items) == 2  # punctuation/case-insensitive title dedupe
    assert items[0]["source"] == "Reuters" and items[0]["url"] == ""


def test_build_pack_indices_align_with_kept_items() -> None:
    items = [
        {"source": "Reuters", "title": f"t{i}", "text": f"text number {i}",
         "url": f"https://e/{i}", "published": pd.Timestamp("2026-06-10", tz="UTC")}
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
    items = [{"source": "S", "title": "t", "text": "a\nb   c" + "y" * 500, "url": "", "published": None}]
    pack, kept = packs.build_pack("NVDA", "Viral / retail", items, [])
    line = next(ln for ln in pack.splitlines() if ln.startswith("[0]"))
    assert "\n" not in line and "a b c" in line
    assert len(line) <= 280 + 40  # text cap + prefix slack


def test_age_suffix_present_when_published_known() -> None:
    items = [{"source": "Reuters", "title": "t", "text": "hello", "url": "",
              "published": pd.Timestamp.now("UTC") - pd.Timedelta(days=2)}]
    pack, _ = packs.build_pack("NVDA", "Official media", items, [])
    assert "(Reuters, 2d)" in pack
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_sentiment_packs.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`tradinglib/sentiment/packs.py`:

```python
"""Bounded per-tier text packs from loader rows.

A pack is a numbered list of one-line items — ``[i] (source, age) text`` —
under a small header carrying the tier's mechanical metrics. The kept-item
list is returned alongside the text so the LLM's ``evidence_indices`` resolve
back to real rows (the model never emits URLs, so links can't be hallucinated).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_ITEM_MAX_CHARS = 280
_PACK_MAX_CHARS = 10_000


def age_days(published: Any) -> float | None:
    """Days since ``published`` (UTC), rounded to 0.1; None when unknown."""
    ts = pd.to_datetime(published, utc=True, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    age = (pd.Timestamp.now("UTC") - ts).total_seconds() / 86_400.0
    return round(max(age, 0.0), 1)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    for item in items:
        key = _norm_title(str(item["title"])) or str(item["text"])[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def news_items(yf_news: pd.DataFrame, google_news: pd.DataFrame) -> list[dict]:
    """Tier-1 items: yfinance headlines (title+summary) + Google News titles."""
    items: list[dict] = []
    for r in yf_news.itertuples():
        text = r.title if not r.summary else f"{r.title} — {r.summary}"
        items.append(
            {"source": r.publisher or "Yahoo Finance", "title": r.title, "text": text,
             "url": "", "published": r.published}
        )
    for r in google_news.itertuples():
        items.append(
            {"source": r.publisher or "Google News", "title": r.title, "text": r.title,
             "url": r.url, "published": r.published}
        )
    return _dedupe(items)


def _reddit_item(r: Any) -> dict:
    text = r.title if not r.text else f"{r.title} — {r.text}"
    return {
        "source": f"r/{r.subreddit} (+{int(r.score)}, {int(r.num_comments)}c)",
        "title": r.title,
        "text": text,
        "url": r.url,
        "published": r.created,
    }


def forum_items(seeking_alpha: pd.DataFrame, reddit_posts: pd.DataFrame) -> list[dict]:
    """Tier-2 items: Seeking Alpha titles + serious-subreddit posts."""
    items: list[dict] = [
        {"source": "Seeking Alpha", "title": r.title, "text": r.title, "url": r.url,
         "published": r.published}
        for r in seeking_alpha.itertuples()
    ]
    items.extend(_reddit_item(r) for r in reddit_posts.itertuples())
    return _dedupe(items)


def viral_items(wsb_posts: pd.DataFrame, stocktwits: pd.DataFrame) -> list[dict]:
    """Tier-3 items: r/wallstreetbets posts + Stocktwits messages."""
    items: list[dict] = [_reddit_item(r) for r in wsb_posts.itertuples()]
    for r in stocktwits.itertuples():
        tag = f" [user-tagged {r.sentiment}]" if r.sentiment else ""
        items.append(
            {"source": "Stocktwits", "title": str(r.body)[:80], "text": f"{r.body}{tag}",
             "url": r.url, "published": r.created}
        )
    return _dedupe(items)


def build_pack(
    ticker: str, tier_label: str, items: list[dict], metric_lines: list[str]
) -> tuple[str, list[dict]]:
    """Assemble the bounded numbered pack; returns ``(pack_text, kept_items)``."""
    header = [f"# {ticker} — {tier_label} sentiment pack", *metric_lines, ""]
    lines = list(header)
    kept: list[dict] = []
    used = sum(len(line) + 1 for line in header)
    for item in items:
        text = " ".join(str(item["text"]).split())[:_ITEM_MAX_CHARS]
        age = age_days(item.get("published"))
        age_part = f", {age:g}d" if age is not None else ""
        line = f"[{len(kept)}] ({item['source']}{age_part}) {text}"
        if used + len(line) + 1 > _PACK_MAX_CHARS:
            break
        lines.append(line)
        kept.append(item)
        used += len(line) + 1
    return "\n".join(lines), kept
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_sentiment_packs.py -q`
Expected: 5 passed
(Note: the age-suffix test asserts `2d` via `:g` formatting of `2.0`. If it renders as `2d` it passes; the rounding in `age_days` keeps one decimal, and `:g` drops the trailing `.0`.)

- [ ] **Step 5: Commit**

```bash
git add tradinglib/sentiment/packs.py tests/test_sentiment_packs.py
git commit -m "feat(sentiment): bounded tier pack assembly"
```

---

### Task 8: Scoring — LLM call + mechanical metrics

**Files:**
- Create: `tradinglib/sentiment/scoring.py`
- Test: `tests/test_sentiment_scoring.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sentiment_scoring.py`:

```python
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
        "score": 0.7, "stance": "bullish", "confidence": 0.8,
        "summary": "Upbeat coverage.", "key_themes": ["dc demand"], "evidence_indices": [0, 2],
    }
    out = scoring.score_tier(_provider("noise " + json.dumps(payload) + " trailing"), "official", "pack")
    assert out["score"] == 0.7 and out["stance"] == "bullish"
    assert out["evidence_indices"] == [0, 2] and out["parse_error"] is False


def test_score_tier_clamps_and_validates() -> None:
    payload = {
        "score": 3.2, "stance": "moonshot", "confidence": -2,
        "summary": "x", "key_themes": [1, 2, 3, 4, 5, 6, 7], "evidence_indices": [0, 0, True, -1, "2", 1, 3, 4, 9],
    }
    out = scoring.score_tier(_provider(payload), "viral", "pack")
    assert out["score"] == 1.0 and out["stance"] is None and out["confidence"] == 0.0
    assert len(out["key_themes"]) == 5
    assert out["evidence_indices"] == [0, 1, 3, 4, 9]  # bools/dupes/negatives/strings dropped, capped at 5


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
        [{"created": "2026-06-10", "score": 10, "num_comments": 4},
         {"created": "2026-06-09", "score": 30, "num_comments": 6}]
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
    m = scoring.viral_metrics(wsb, st, interest)
    assert m["wsb_mentions"] == 1 and m["st_messages"] == 4
    assert m["st_bullish"] == 2 and m["st_bearish"] == 1
    assert m["st_bull_bear_ratio"] == 2.0
    assert m["trends_spike"] is not None and m["trends_spike"] > 2.0


def test_viral_metrics_guards() -> None:
    st = pd.DataFrame({"sentiment": ["Bullish"]})
    m = scoring.viral_metrics(pd.DataFrame(), st, None)
    assert m["st_bull_bear_ratio"] is None  # no bears -> undefined, not inf
    assert m["trends_spike"] is None


def test_trends_spike_zero_baseline_is_none() -> None:
    idx = pd.date_range(end=pd.Timestamp.now("UTC"), periods=20, freq="D")
    interest = pd.Series([0.0] * 13 + [50.0] * 7, index=idx)
    assert scoring.trends_spike(interest) is None


def test_official_metrics() -> None:
    assert scoring.official_metrics(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1, 2]})) == {
        "headline_count": 3
    }
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_sentiment_scoring.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`tradinglib/sentiment/scoring.py`:

```python
"""Per-tier LLM scoring + mechanical (no-LLM) metrics.

``score_tier`` mirrors ``scanner/briefs.py``: ONE provider call demanding
strict JSON, hard sanitization, graceful degradation (malformed JSON keeps the
raw text as the summary with ``score=None``) — it never raises. Mechanical
metrics are plain-code facts computed from loader rows, independent of the LLM.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from tradinglib.assistant.provider import LLMProvider
from tradinglib.assistant.types import UserMsg
from tradinglib.sentiment.types import TIER_FORUMS, TIER_OFFICIAL, TIER_VIRAL

_STANCES = ("bearish", "neutral", "bullish", "mixed")
_MAX_THEMES = 5
_MAX_EVIDENCE = 5
_RAW_SUMMARY_MAX_CHARS = 600

_TIER_DESC = {
    TIER_OFFICIAL: "Tier-1 financial news headlines",
    TIER_FORUMS: "serious investor forums (Seeking Alpha article titles, finance subreddits)",
    TIER_VIRAL: "viral retail chatter (r/wallstreetbets posts, Stocktwits messages)",
}

SCORE_SYSTEM_PROMPT = (
    "You are a market-sentiment analyst. You receive a pack of recent text items "
    "about one stock, all from a single source tier — {tier_desc}. Items are "
    "numbered like [3]; the header lists mechanical metrics for context. Judge "
    "the aggregate sentiment of the pack toward the stock over the coming weeks. "
    "Respond with ONLY a JSON object, no prose before or after, with exactly "
    "these keys: score (number, -1 strongly bearish to 1 strongly bullish), "
    "stance (one of: bearish, neutral, bullish, mixed — mixed means genuinely "
    "conflicting reads, not merely thin data), confidence (number 0-1: how "
    "substantive and consistent the pack is), summary (string, 2-3 sentences), "
    "key_themes (array of at most 5 short strings), evidence_indices (array of "
    "at most 5 integers — the item numbers that most drove your judgement). "
    "Ground every claim in the pack; do not invent facts."
)


def _clamp(value: Any, lo: float, hi: float) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return min(max(f, lo), hi)


def _sanitize(raw: dict, *, parse_error: bool = False) -> dict:
    stance = raw.get("stance")
    if stance not in _STANCES:
        stance = None
    indices: list[int] = []
    for x in raw.get("evidence_indices") or []:
        if isinstance(x, int) and not isinstance(x, bool) and x >= 0 and x not in indices:
            indices.append(x)
    return {
        "score": _clamp(raw.get("score"), -1.0, 1.0),
        "stance": stance,
        "confidence": _clamp(raw.get("confidence"), 0.0, 1.0),
        "summary": str(raw.get("summary", "")),
        "key_themes": [str(x) for x in (raw.get("key_themes") or [])][:_MAX_THEMES],
        "evidence_indices": indices[:_MAX_EVIDENCE],
        "parse_error": parse_error,
    }


def empty_scoring() -> dict:
    """Neutral scoring shell for tiers with no content (no LLM call made)."""
    return _sanitize({})


def unavailable(reason: str) -> dict:
    """Scoring stand-in when the LLM cannot run (no key, provider down)."""
    return _sanitize({"summary": reason}, parse_error=True)


def score_tier(provider: LLMProvider, tier: str, pack: str) -> dict:
    """One provider call -> sanitized tier-scoring dict; never raises."""
    system = SCORE_SYSTEM_PROMPT.format(tier_desc=_TIER_DESC[tier])
    try:
        turn = provider.complete(system, [UserMsg(pack)], tools=[])
    except Exception as exc:
        return unavailable(f"LLM scoring unavailable: {exc}")
    text = turn.text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return _sanitize(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    return _sanitize({"summary": text[:_RAW_SUMMARY_MAX_CHARS]}, parse_error=True)


def official_metrics(yf_news: pd.DataFrame, google_news: pd.DataFrame) -> dict:
    return {"headline_count": int(len(yf_news) + len(google_news))}


def forum_metrics(seeking_alpha: pd.DataFrame, reddit_posts: pd.DataFrame) -> dict:
    metrics: dict[str, Any] = {"post_count": int(len(seeking_alpha) + len(reddit_posts))}
    if len(reddit_posts) > 0:
        metrics["mean_upvotes"] = round(float(reddit_posts["score"].mean()), 1)
        metrics["mean_comments"] = round(float(reddit_posts["num_comments"].mean()), 1)
    else:
        metrics["mean_upvotes"] = None
        metrics["mean_comments"] = None
    return metrics


def trends_spike(interest: pd.Series | None) -> float | None:
    """mean(last 7 days) / mean(prior window); None when either side is missing."""
    if interest is None or len(interest) == 0:
        return None
    cutoff = interest.index.max() - pd.Timedelta(days=7)
    recent = interest[interest.index > cutoff]
    base = interest[interest.index <= cutoff]
    if len(recent) == 0 or len(base) == 0 or float(base.mean()) <= 0.0:
        return None
    return round(float(recent.mean()) / float(base.mean()), 2)


def viral_metrics(
    wsb_posts: pd.DataFrame, stocktwits: pd.DataFrame, interest: pd.Series | None
) -> dict:
    bulls = int((stocktwits["sentiment"] == "Bullish").sum()) if len(stocktwits) else 0
    bears = int((stocktwits["sentiment"] == "Bearish").sum()) if len(stocktwits) else 0
    return {
        "wsb_mentions": int(len(wsb_posts)),
        "st_messages": int(len(stocktwits)),
        "st_bullish": bulls,
        "st_bearish": bears,
        "st_bull_bear_ratio": round(bulls / bears, 2) if bears > 0 else None,
        "trends_spike": trends_spike(interest),
    }
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_sentiment_scoring.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add tradinglib/sentiment/scoring.py tests/test_sentiment_scoring.py
git commit -m "feat(sentiment): llm tier scoring + mechanical metrics"
```

---

### Task 9: Orchestrator — `run_sentiment`

**Files:**
- Create: `tradinglib/sentiment/report.py`
- Modify: `tradinglib/sentiment/__init__.py`
- Test: `tests/test_sentiment_report.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sentiment_report.py`:

```python
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
        "score": score, "stance": stance, "confidence": 0.8,
        "summary": f"{stance} read.", "key_themes": ["theme"], "evidence_indices": [0],
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


def test_provider_construction_failure_degrades(stubbed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_key() -> None:
        raise RuntimeError("ANTHROPIC_API_KEY missing")

    monkeypatch.setattr(report_mod, "ClaudeProvider", _no_key)
    rep = report_mod.run_sentiment("NVDA")  # provider=None -> tries ClaudeProvider()
    assert rep.status == "partial"
    for tier in rep.tiers:
        assert tier.status == "degraded" and "unavailable" in tier.summary
        assert tier.metrics  # mechanical metrics still present
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_sentiment_report.py -q`
Expected: FAIL — `report` module not found

- [ ] **Step 3: Implement**

`tradinglib/sentiment/report.py`:

```python
"""``run_sentiment`` — fetch -> pack -> score -> aggregate for one ticker.

Sources fetch concurrently (thread pool); each failure is recorded per source
and its tier proceeds on whatever remains. Non-empty tiers are scored
concurrently (one provider call each). Aggregation is deterministic: overall
bias is the mean of available tier scores, and a divergence callout fires when
two tiers disagree by >= ``DIVERGENCE_GAP``. The finished report is cached to
``data/processed/sentiment/reports/<ticker>/<date>.json`` — same-day lookups
are served from that JSON (``refresh=True`` bypasses), and the directory
accrues the forward history a future nightly batch job needs.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any

import pandas as pd

from tradinglib.assistant.provider import ClaudeProvider, LLMProvider
from tradinglib.data.paths import processed_dir
from tradinglib.loaders.forums.reddit import get_reddit_posts
from tradinglib.loaders.forums.seeking_alpha import get_seeking_alpha
from tradinglib.loaders.news.google_news import get_google_news
from tradinglib.loaders.news.yfinance import get_news
from tradinglib.loaders.sentiment.google_trends import load_interest
from tradinglib.loaders.social.stocktwits import get_stocktwits
from tradinglib.sentiment import packs, scoring
from tradinglib.sentiment.types import (
    DIVERGENCE_GAP,
    TIER_FORUMS,
    TIER_LABELS,
    TIER_OFFICIAL,
    TIER_VIRAL,
    Evidence,
    SentimentReport,
    TierReport,
)

SOURCE = "sentiment"
_REPORTS = "reports"
SERIOUS_SUBREDDITS = ("stocks", "investing", "ValueInvesting", "SecurityAnalysis")
VIRAL_SUBREDDITS = ("wallstreetbets",)
_TRENDS_WINDOW_DAYS = 97  # 7-day spike window vs ~90-day baseline

logger = logging.getLogger(__name__)


def _trends_series(ticker: str, *, refresh: bool) -> pd.Series:
    now = pd.Timestamp.now("UTC")
    start = (now - pd.Timedelta(days=_TRENDS_WINDOW_DAYS)).strftime("%Y-%m-%d")
    return load_interest(
        f"{ticker} stock", timeframe=f"{start} {now.strftime('%Y-%m-%d')}", refresh=refresh
    )


def _fetch_sources(ticker: str, *, refresh: bool) -> tuple[dict[str, Any], dict[str, str]]:
    jobs = {
        "yfinance_news": lambda: get_news(ticker, max_items=15, refresh=refresh),
        "google_news": lambda: get_google_news(ticker, max_items=25, refresh=refresh),
        "seeking_alpha": lambda: get_seeking_alpha(ticker, max_items=20, refresh=refresh),
        "reddit": lambda: get_reddit_posts(
            ticker, SERIOUS_SUBREDDITS, limit=10, refresh=refresh
        ).head(20),
        "wsb": lambda: get_reddit_posts(
            ticker, VIRAL_SUBREDDITS, limit=20, refresh=refresh
        ).head(20),
        "stocktwits": lambda: get_stocktwits(ticker, max_items=30, refresh=refresh),
        "google_trends": lambda: _trends_series(ticker, refresh=refresh),
    }
    data: dict[str, Any] = {}
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, future in futures.items():
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("sentiment source %s failed for %s: %s", name, ticker, exc)
                data[name] = None
                status[name] = f"error: {exc}"
                continue
            data[name] = result
            status[name] = "ok" if len(result) > 0 else "empty"
    return data, status


def _frame(data: dict[str, Any], key: str) -> pd.DataFrame:
    value = data.get(key)
    return value if value is not None else pd.DataFrame()


def _tier_status(item_count: int, score: dict, source_status: dict[str, str]) -> str:
    if item_count == 0:
        return "no_data"
    if score["parse_error"] or score["score"] is None:
        return "degraded"
    if any(v.startswith("error") for v in source_status.values()):
        return "degraded"
    return "ok"


def _evidence(score: dict, kept: list[dict]) -> list[Evidence]:
    out: list[Evidence] = []
    for i in score["evidence_indices"]:
        if i < len(kept):
            item = kept[i]
            out.append(
                Evidence(
                    title=str(item["title"])[:160],
                    source=str(item["source"]),
                    url=str(item.get("url", "")),
                    age_days=packs.age_days(item.get("published")),
                )
            )
    return out


def _divergence(tier_scores: dict[str, float]) -> dict[str, Any] | None:
    if len(tier_scores) < 2:
        return None
    hi = max(tier_scores, key=lambda k: tier_scores[k])
    lo = min(tier_scores, key=lambda k: tier_scores[k])
    gap = round(tier_scores[hi] - tier_scores[lo], 3)
    if gap < DIVERGENCE_GAP:
        return None
    return {"pair": [hi, lo], "gap": gap}


def run_sentiment(
    ticker: str, *, refresh: bool = False, provider: LLMProvider | None = None
) -> SentimentReport:
    """Three-tier sentiment read for one ticker; cached per (ticker, UTC day)."""
    ticker = ticker.strip().upper()
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _REPORTS / ticker / f"{snapshot}.json"
    if out.exists() and not refresh:
        return SentimentReport.from_dict(json.loads(out.read_text(encoding="utf-8")))

    data, status = _fetch_sources(ticker, refresh=refresh)
    trends = data.get("google_trends")  # Series or None

    tiers_def = [
        (
            TIER_OFFICIAL,
            packs.news_items(_frame(data, "yfinance_news"), _frame(data, "google_news")),
            scoring.official_metrics(_frame(data, "yfinance_news"), _frame(data, "google_news")),
            {k: status[k] for k in ("yfinance_news", "google_news")},
        ),
        (
            TIER_FORUMS,
            packs.forum_items(_frame(data, "seeking_alpha"), _frame(data, "reddit")),
            scoring.forum_metrics(_frame(data, "seeking_alpha"), _frame(data, "reddit")),
            {k: status[k] for k in ("seeking_alpha", "reddit")},
        ),
        (
            TIER_VIRAL,
            packs.viral_items(_frame(data, "wsb"), _frame(data, "stocktwits")),
            scoring.viral_metrics(_frame(data, "wsb"), _frame(data, "stocktwits"), trends),
            {k: status[k] for k in ("wsb", "stocktwits", "google_trends")},
        ),
    ]

    built: dict[str, dict[str, Any]] = {}
    for tier_id, items, metrics, src_status in tiers_def:
        metric_lines = [f"{k}: {'n/a' if v is None else v}" for k, v in metrics.items()]
        pack, kept = packs.build_pack(ticker, TIER_LABELS[tier_id], items, metric_lines)
        built[tier_id] = {"pack": pack, "kept": kept, "metrics": metrics, "src": src_status}

    to_score = [tier_id for tier_id, b in built.items() if b["kept"]]
    llm_error: str | None = None
    if to_score and provider is None:
        try:
            provider = ClaudeProvider()
        except Exception as exc:
            llm_error = f"LLM scoring unavailable: {exc}"

    score_results: dict[str, dict] = {}
    if to_score and provider is not None:
        with ThreadPoolExecutor(max_workers=len(to_score)) as pool:
            futures = {
                tier_id: pool.submit(scoring.score_tier, provider, tier_id, built[tier_id]["pack"])
                for tier_id in to_score
            }
            score_results = {tier_id: f.result() for tier_id, f in futures.items()}
    for tier_id in to_score:
        if tier_id not in score_results:
            score_results[tier_id] = scoring.unavailable(llm_error or "LLM scoring unavailable")

    tier_reports: list[TierReport] = []
    for tier_id, _items, _metrics, _src in tiers_def:
        b = built[tier_id]
        s = score_results.get(tier_id, scoring.empty_scoring())
        tier_reports.append(
            TierReport(
                tier=tier_id,
                label=TIER_LABELS[tier_id],
                status=_tier_status(len(b["kept"]), s, b["src"]),
                score=s["score"],
                stance=s["stance"],
                confidence=s["confidence"],
                summary=s["summary"],
                key_themes=s["key_themes"],
                evidence=_evidence(s, b["kept"]),
                metrics=b["metrics"],
                source_status=b["src"],
                item_count=len(b["kept"]),
                parse_error=s["parse_error"],
            )
        )

    tier_scores = {t.tier: t.score for t in tier_reports if t.score is not None}
    statuses = {t.status for t in tier_reports}
    report = SentimentReport(
        ticker=ticker,
        as_of=pd.Timestamp.now("UTC").isoformat(),
        status="no_data" if statuses == {"no_data"} else ("ok" if statuses == {"ok"} else "partial"),
        tiers=tier_reports,
        overall_bias=round(mean(tier_scores.values()), 3) if tier_scores else None,
        divergence=_divergence(tier_scores),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
```

Replace `tradinglib/sentiment/__init__.py` with:

```python
"""Three-tier ticker sentiment engine (official / forums / viral)."""

from tradinglib.sentiment.report import run_sentiment
from tradinglib.sentiment.types import SentimentReport, TierReport

__all__ = ["SentimentReport", "TierReport", "run_sentiment"]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_sentiment_report.py tests/test_sentiment_types.py tests/test_sentiment_packs.py tests/test_sentiment_scoring.py -q`
Expected: all pass (the `__init__.py` change must not break the earlier sentiment tests)

- [ ] **Step 5: Commit**

```bash
git add tradinglib/sentiment/report.py tradinglib/sentiment/__init__.py tests/test_sentiment_report.py
git commit -m "feat(sentiment): run_sentiment orchestrator with caching + divergence"
```

---

### Task 10: Webapp — `/sentiment` page + API

**Files:**
- Create: `webapp/sentiment.py`, `webapp/templates/sentiment.html`
- Modify: `webapp/main.py` (import + 2 routes, after the `/planner` route)
- Modify: `webapp/templates/scans.html`, `planner.html`, `tournaments.html`, `tournament_day.html`, `models.html`, `index.html` (nav crumb)
- Test: `tests/test_webapp_sentiment.py`

- [ ] **Step 1: Write failing tests**

`tests/test_webapp_sentiment.py`:

```python
"""Tests for the /sentiment page and API route (engine stubbed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp.main import create_app


def test_sentiment_page_renders() -> None:
    resp = TestClient(create_app()).get("/sentiment")
    assert resp.status_code == 200
    assert "sentiment" in resp.text.lower()


def test_sentiment_api_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake(ticker: str, *, refresh: bool = False) -> dict:
        captured["ticker"] = ticker
        captured["refresh"] = refresh
        return {"ticker": ticker.upper(), "status": "ok", "tiers": []}

    monkeypatch.setattr("webapp.sentiment.get_report", _fake)
    resp = TestClient(create_app()).get("/api/v1/sentiment/nvda?refresh=1")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "NVDA"
    assert captured == {"ticker": "nvda", "refresh": True}


def test_sentiment_api_invalid_ticker_400() -> None:
    resp = TestClient(create_app()).get("/api/v1/sentiment/NOT%20A%20TICKER!!")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_sentiment_api_engine_failure_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ticker: str, *, refresh: bool = False) -> dict:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("webapp.sentiment.get_report", _boom)
    resp = TestClient(create_app()).get("/api/v1/sentiment/NVDA")
    assert resp.status_code == 500
    assert "error" in resp.json()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_webapp_sentiment.py -q`
Expected: FAIL — 404s / missing module

- [ ] **Step 3: Implement view helper + routes**

`webapp/sentiment.py`:

```python
"""View helper for the /sentiment page — ticker validation + engine call."""

from __future__ import annotations

import re

_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


def valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_RE.match(ticker.strip()))


def get_report(ticker: str, *, refresh: bool = False) -> dict:
    from tradinglib.sentiment.report import run_sentiment  # lazy: keeps webapp import light

    return run_sentiment(ticker, refresh=refresh).to_dict()
```

In `webapp/main.py`, add to the webapp import block (after `from webapp import scans as _scans`):

```python
from webapp import sentiment as _sentiment
```

and add the routes inside `create_app()` directly after the `/planner` route:

```python
    @app.get("/sentiment", response_class=HTMLResponse)
    def sentiment_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "sentiment.html", {})

    @app.get("/api/v1/sentiment/{ticker}")
    def sentiment_api(ticker: str, refresh: bool = False) -> JSONResponse:
        if not _sentiment.valid_ticker(ticker):
            return JSONResponse({"error": "invalid ticker"}, status_code=400)
        try:
            return JSONResponse(_sentiment.get_report(ticker, refresh=refresh))
        except Exception as exc:  # engine degrades internally; this is the last resort
            return JSONResponse({"error": str(exc)}, status_code=500)
```

- [ ] **Step 4: Create the template**

`webapp/templates/sentiment.html` (theme scaffold — pre-paint script, fonts, `:root` variables, topbar/crumb/toggle CSS and the theme-toggle script — copied from `scans.html` so the page matches the house look):

```html
<!doctype html>
<html lang="en" data-theme="bone">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trading Models — Ticker Sentiment</title>

  <!-- set theme before first paint to avoid a flash -->
  <script>
    (function () {
      try { document.documentElement.setAttribute("data-theme", localStorage.getItem("tm-theme") || "bone"); }
      catch (e) { document.documentElement.setAttribute("data-theme", "bone"); }
    })();
  </script>

  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Martian+Mono:wght@500;600&display=swap" rel="stylesheet" />

  <style>
    :root[data-theme="bone"] {
      --bg:#e9e4d6; --panel:#ded8c7; --inset:#efe9db;
      --ink:#1c1b16; --muted:#6f6a5c; --line:#c7bfa9; --line-strong:#1c1b16;
      --up:#1f7a3d; --down:#b3261e; --accent:#c8852f; --accent-ink:#1c1b16;
    }
    :root[data-theme="night"] {
      --bg:#121110; --panel:#1a1815; --inset:#171511;
      --ink:#e8e2d4; --muted:#8a8478; --line:#34302a; --line-strong:#4a463e;
      --up:#3fbf6b; --down:#e5534b; --accent:#d39a4f; --accent-ink:#121110;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--ink);
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
      font-size: 13px; line-height: 1.5;
      transition: background .18s ease, color .18s ease;
    }
    .label, .brand, .m-label, .chip, .crumb { font-family: "Martian Mono", "JetBrains Mono", monospace;
      text-transform: uppercase; letter-spacing: .12em; }

    .topbar { display: flex; align-items: center; gap: 14px;
              padding: 11px 18px; border-bottom: 1px solid var(--line-strong); }
    .brand { font-size: 14px; font-weight: 600; letter-spacing: .18em; }
    .brand a { color: inherit; text-decoration: none; }
    .crumb { font-size: 10px; color: var(--muted); letter-spacing: .18em; }
    .spacer { flex: 1; }
    .toggle { display: flex; border: 1px solid var(--line-strong); }
    .toggle button { font-family: "Martian Mono", monospace; font-size: 9px; letter-spacing: .14em;
      background: transparent; color: var(--muted); border: 0; padding: 6px 11px;
      cursor: pointer; text-transform: uppercase; }
    .toggle button[aria-pressed="true"] { background: var(--ink); color: var(--bg); }

    .wrap { max-width: 980px; margin: 0 auto; padding: 22px 18px 60px; }
    .placeholder { color: var(--muted); }

    .lookup { display: flex; gap: 8px; margin: 0 0 18px; }
    .lookup input { font: inherit; text-transform: uppercase; background: var(--inset);
      color: var(--ink); border: 1px solid var(--line-strong); padding: 8px 10px; width: 140px; }
    .lookup button { font-family: "Martian Mono", monospace; font-size: 10px; letter-spacing: .12em;
      text-transform: uppercase; background: var(--ink); color: var(--bg);
      border: 1px solid var(--line-strong); padding: 8px 14px; cursor: pointer; }
    .lookup button.secondary { background: transparent; color: var(--muted); }
    .lookup button:disabled { opacity: .5; cursor: wait; }

    .overall { display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
      border: 1px solid var(--line-strong); padding: 12px 14px; margin-bottom: 18px; }
    .overall .m-label { font-size: 8.5px; color: var(--muted); }
    .overall .m-value { font-size: 22px; font-weight: 700; }
    .divergence { border: 1px solid var(--accent); color: var(--accent-ink);
      background: var(--accent); padding: 6px 10px; font-size: 11px; }

    .tiers { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    @media (max-width: 860px) { .tiers { grid-template-columns: 1fr; } }
    .card { border: 1px solid var(--line-strong); background: var(--panel); padding: 12px 14px; }
    .card h3 { margin: 0 0 8px; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; }
    .badge { display: inline-block; font-size: 9px; letter-spacing: .12em; text-transform: uppercase;
      padding: 3px 8px; border: 1px solid var(--line-strong); margin-left: 8px; }
    .badge.bullish { color: var(--up); border-color: var(--up); }
    .badge.bearish { color: var(--down); border-color: var(--down); }
    .badge.mixed { color: var(--accent); border-color: var(--accent); }
    .badge.neutral, .badge.na { color: var(--muted); }

    .dial { position: relative; height: 8px; background: var(--inset);
      border: 1px solid var(--line); margin: 10px 0 4px; }
    .dial .tick { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px;
      background: var(--muted); }
    .dial .marker { position: absolute; top: -4px; width: 9px; height: 14px;
      transform: translateX(-50%); background: var(--muted); }
    .dial .marker.pos { background: var(--up); }
    .dial .marker.neg { background: var(--down); }
    .dial-scale { display: flex; justify-content: space-between; font-size: 9px;
      color: var(--muted); margin-bottom: 8px; }

    .metrics { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
    .chip { font-size: 8.5px; color: var(--muted); border: 1px solid var(--line);
      padding: 3px 7px; }
    .chip b { color: var(--ink); }

    .summary { margin: 8px 0; }
    .themes { color: var(--muted); font-size: 11px; margin: 6px 0; }
    .evidence { margin: 8px 0 0; padding: 0; list-style: none; font-size: 11px; }
    .evidence li { margin: 4px 0; }
    .evidence a { color: inherit; }
    .src-note { font-size: 9.5px; color: var(--muted); border-top: 1px solid var(--line);
      margin-top: 10px; padding-top: 6px; }
    .error { color: var(--down); }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand"><a href="/">TRADING MODELS</a></div>
    <div class="crumb">/ sentiment</div>
    <div class="crumb"><a href="/scans" style="color: inherit; text-decoration: none;">/ swing scans</a></div>
    <div class="crumb"><a href="/tournaments" style="color: inherit; text-decoration: none;">/ tournaments</a></div>
    <div class="crumb"><a href="/planner" style="color: inherit; text-decoration: none;">/ options planner</a></div>
    <div class="spacer"></div>
    <div class="toggle" role="group" aria-label="theme">
      <button type="button" data-theme-set="bone">Bone</button>
      <button type="button" data-theme-set="night">Night</button>
    </div>
  </div>

  <div class="wrap">
    <form class="lookup" id="lookupForm">
      <input id="ticker" placeholder="TICKER" maxlength="10" autocomplete="off" autofocus />
      <button type="submit" id="readBtn">Read sentiment</button>
      <button type="button" id="refreshBtn" class="secondary" title="Bypass today's cache and refetch every source">Re-fetch</button>
    </form>

    <p class="placeholder" id="status">Three-tier read — official media vs serious forums vs viral retail.
      First lookup of the day fetches live sources and takes ~10–20s.</p>

    <div id="out" style="display: none;">
      <div class="overall">
        <div>
          <div class="m-label">Overall bias</div>
          <div class="m-value" id="overallBias">—</div>
        </div>
        <div>
          <div class="m-label">As of</div>
          <div id="asOf" style="font-size: 11px;">—</div>
        </div>
        <div class="divergence" id="divergence" style="display: none;"></div>
      </div>
      <div class="tiers" id="tierGrid"></div>
    </div>
  </div>

  <script>
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
      (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
    const fmtScore = (v) => (v === null || v === undefined) ? "—" : (v > 0 ? "+" : "") + v.toFixed(2);
    const TIER_NAMES = {official: "Official media", forums: "Serious forums", viral: "Viral / retail"};

    function dial(score) {
      if (score === null || score === undefined) return "";
      const pct = ((score + 1) / 2) * 100;
      const cls = score > 0.05 ? "pos" : (score < -0.05 ? "neg" : "");
      return `<div class="dial"><div class="tick"></div><div class="marker ${cls}" style="left: ${pct}%"></div></div>
        <div class="dial-scale"><span>-1 bearish</span><span>0</span><span>+1 bullish</span></div>`;
    }

    function metricChips(metrics) {
      return Object.entries(metrics || {}).map(([k, v]) =>
        `<span class="chip">${esc(k.replaceAll("_", " "))}: <b>${v === null ? "n/a" : esc(v)}</b></span>`
      ).join("");
    }

    function evidenceList(evidence) {
      if (!evidence || !evidence.length) return "";
      const rows = evidence.map((e) => {
        const age = e.age_days === null || e.age_days === undefined ? "" : `, ${e.age_days}d`;
        const body = `${esc(e.title)} <span class="placeholder">(${esc(e.source)}${age})</span>`;
        return `<li>${e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">${body}</a>` : body}</li>`;
      });
      return `<ul class="evidence">${rows.join("")}</ul>`;
    }

    function tierCard(t) {
      const stance = t.stance || (t.status === "no_data" ? "na" : "neutral");
      const conf = t.confidence === null || t.confidence === undefined ? "" :
        `<span class="chip">confidence: <b>${t.confidence.toFixed(2)}</b></span>`;
      const note = Object.entries(t.source_status || {}).map(([k, v]) =>
        `${esc(k)}: ${esc(v)}`).join(" · ");
      const body = t.status === "no_data"
        ? `<p class="placeholder">No recent items found.</p>`
        : `${dial(t.score)}
           <div class="metrics">${metricChips(t.metrics)}${conf}</div>
           <p class="summary">${esc(t.summary)}</p>
           ${t.key_themes && t.key_themes.length ? `<div class="themes">themes: ${esc(t.key_themes.join(" · "))}</div>` : ""}
           ${evidenceList(t.evidence)}`;
      return `<div class="card">
        <h3>${esc(t.label)} <span class="badge ${esc(stance)}">${esc(t.status === "no_data" ? "no data" : (t.stance || "unscored"))}</span>
          ${t.score !== null && t.score !== undefined ? `<span style="float:right">${fmtScore(t.score)}</span>` : ""}</h3>
        ${body}
        <div class="src-note">${note}${t.status === "degraded" ? ' · <span class="error">degraded</span>' : ""}</div>
      </div>`;
    }

    function render(report) {
      document.getElementById("overallBias").textContent = fmtScore(report.overall_bias);
      document.getElementById("asOf").textContent = report.as_of || "—";
      const div = document.getElementById("divergence");
      if (report.divergence) {
        div.style.display = "";
        div.textContent = `divergence: ${TIER_NAMES[report.divergence.pair[0]] || report.divergence.pair[0]} vs ` +
          `${TIER_NAMES[report.divergence.pair[1]] || report.divergence.pair[1]} — gap ${report.divergence.gap.toFixed(2)}`;
      } else {
        div.style.display = "none";
      }
      document.getElementById("tierGrid").innerHTML = (report.tiers || []).map(tierCard).join("");
      document.getElementById("out").style.display = "";
    }

    async function lookup(refresh) {
      const ticker = document.getElementById("ticker").value.trim().toUpperCase();
      const status = document.getElementById("status");
      if (!/^[A-Z][A-Z0-9.\-]{0,9}$/.test(ticker)) {
        status.textContent = "Enter a ticker (e.g. NVDA).";
        return;
      }
      const buttons = [document.getElementById("readBtn"), document.getElementById("refreshBtn")];
      buttons.forEach((b) => (b.disabled = true));
      status.textContent = refresh
        ? `Re-fetching every source for ${ticker}… (~10–20s)`
        : `Reading ${ticker}… first lookup of the day fetches live sources (~10–20s).`;
      try {
        const resp = await fetch(`/api/v1/sentiment/${ticker}${refresh ? "?refresh=1" : ""}`);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        render(data);
        status.textContent = data.status === "no_data"
          ? `Nothing found for ${ticker} in any tier — no fake neutral here.`
          : `${ticker} — report status: ${data.status}.`;
      } catch (err) {
        status.innerHTML = `<span class="error">Lookup failed: ${esc(err.message)}</span>`;
      } finally {
        buttons.forEach((b) => (b.disabled = false));
      }
    }

    document.getElementById("lookupForm").addEventListener("submit", (e) => { e.preventDefault(); lookup(false); });
    document.getElementById("refreshBtn").addEventListener("click", () => lookup(true));

    (function () {
      const buttons = document.querySelectorAll("[data-theme-set]");
      function apply(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        try { localStorage.setItem("tm-theme", theme); } catch (e) {}
        buttons.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.themeSet === theme)));
      }
      buttons.forEach(b => b.addEventListener("click", () => apply(b.dataset.themeSet)));
      apply(document.documentElement.getAttribute("data-theme") || "bone");
    })();
  </script>
</body>
</html>
```

- [ ] **Step 5: Add the nav crumb to the other pages**

Run `grep -n '"/planner"' webapp/templates/*.html` to find each page's crumb row. In every template that links to `/planner` in its topbar (`scans.html`, `tournaments.html`, `tournament_day.html`, `models.html`, `index.html` — and in `planner.html`, which links to the *other* pages), add this line alongside the existing crumb links:

```html
    <div class="crumb"><a href="/sentiment" style="color: inherit; text-decoration: none;">/ sentiment</a></div>
```

(Skip any template that has no topbar crumbs, e.g. partials like `_results.html`/`_console.html`.)

- [ ] **Step 6: Run tests, verify pass**

Run: `uv run pytest tests/test_webapp_sentiment.py tests/test_webapp_routes.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add webapp/sentiment.py webapp/templates/ webapp/main.py tests/test_webapp_sentiment.py
git commit -m "feat(webapp): /sentiment page - three-tier ticker sentiment read"
```

---

### Task 11: Docs — data sources + deploy env

**Files:**
- Modify: `docs/data-sources.md` (append under "Currently wired in", matching the existing entry format)
- Modify: `docs/DEPLOY.md` (Reddit env vars next to the existing `ANTHROPIC_API_KEY` mentions)

- [ ] **Step 1: Append the four source entries to `docs/data-sources.md`**

```markdown
### Google News RSS — ticker headlines

- **What**: Recent news headlines per ticker (`[ticker, published, title,
  publisher, url]`), query `"<TICKER> stock when:14d"`.
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.news.google_news`](../tradinglib/loaders/news/google_news.py)
- **Notes**: Public RSS endpoint; mocked in tests. Snapshot-cached per UTC day.
  Tier-1 source for the `/sentiment` page.

### Seeking Alpha RSS — per-ticker article titles

- **What**: Article/analysis titles per ticker (`[ticker, published, title, url]`).
- **Cost**: Free.
- **Setup**: None — no API key required.
- **Loader**: [`tradinglib.loaders.forums.seeking_alpha`](../tradinglib/loaders/forums/seeking_alpha.py)
- **Notes**: Seeking Alpha's public RSS feed (titles only — no bodies, no API).
  The most fragile sentiment source (Cloudflare moods); failures degrade to
  empty and Tier 2 proceeds on Reddit alone.

### Reddit — forum posts mentioning a ticker

- **What**: Posts from configurable subreddits (`[ticker, subreddit, created,
  title, text, score, num_comments, url]`), last week, search `"<T> OR $<T>"`.
- **Cost**: Free (OAuth app).
- **Setup**: Create a **script** app at <https://www.reddit.com/prefs/apps>; set
  `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` (and optionally
  `REDDIT_USER_AGENT`). Without credentials the sentiment engine skips Reddit
  sources gracefully.
- **Loader**: [`tradinglib.loaders.forums.reddit`](../tradinglib/loaders/forums/reddit.py)
- **Notes**: praw client, cached per (subreddit, ticker, day) — cache hits need
  no credentials. Serves Tier 2 (serious subs) and Tier 3 (r/wallstreetbets).

### Stocktwits — retail message stream

- **What**: Last ~30 messages per symbol with user-tagged Bullish/Bearish labels
  (`[ticker, created, body, sentiment, username, url]`).
- **Cost**: Free.
- **Setup**: None — no API key required (~200 requests/hour/IP).
- **Loader**: [`tradinglib.loaders.social.stocktwits`](../tradinglib/loaders/social/stocktwits.py)
- **Notes**: The user tags feed the mechanical bull/bear ratio on the
  `/sentiment` page — free ground truth, no LLM involved.
```

- [ ] **Step 2: Document the env vars in `docs/DEPLOY.md`**

Where the Modal secret is created (the `uv run modal secret create trading-models-secrets ANTHROPIC_API_KEY=...` line), extend the example:

```bash
uv run modal secret create trading-models-secrets ANTHROPIC_API_KEY=sk-ant-... \
  REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...
```

In the Render environment section (the `ANTHROPIC_API_KEY` bullet), add:

```markdown
   - `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` = Reddit script-app credentials
     (optional — the /sentiment page's Reddit sources degrade gracefully without
     them; create the app at reddit.com/prefs/apps).
```

- [ ] **Step 3: Commit**

```bash
git add docs/data-sources.md docs/DEPLOY.md
git commit -m "docs: sentiment data sources + reddit env vars"
```

---

### Task 12: Full verification gate

- [ ] **Step 1: Run the CI gate locally**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy tradinglib
uv run pytest -q
```

Expected: all clean (≈1180 pre-existing tests + ~35 new). If `ruff format --check` complains, run `uv run ruff format .` and re-run the gate. Then run the remaining CI steps exactly as `.github/workflows/ci.yml` defines them (Streamlit import check, MODELS.md freshness) — both should be untouched by this feature.

- [ ] **Step 2 (optional, live-network smoke):** only if `ANTHROPIC_API_KEY` is set locally; Reddit creds optional (tiers degrade):

```bash
uv run python -c "from tradinglib.sentiment.report import run_sentiment; r = run_sentiment('NVDA'); print(r.status, r.overall_bias, [(t.tier, t.status, t.score) for t in r.tiers])"
```

Expected: a `partial`/`ok` report printing three tier tuples; a JSON file appears under `data/processed/sentiment/reports/NVDA/`.

- [ ] **Step 3: Commit any gate fixes**

```bash
git add -A && git commit -m "chore: verification gate fixes"  # only if the gate changed files
```

---

## Plan self-review notes (already applied)

- Spec coverage: loaders (Tasks 2–5), packs/scoring/types/report (6–9), webapp (10), caching + report JSON (9), degradation paths (5, 8, 9), divergence + aggregate (9), docs/env (11), deps (1), tests throughout. Non-goals untouched.
- `evidence_indices` resolve server-side in `report._evidence` — the LLM never emits URLs (spec requirement).
- Missing Reddit creds: loader raises `MissingRedditCredentials` → `_fetch_sources` catches per-source → tier degrades (spec: "soft degrade, not an error"; the engine catches, nothing crashes).
- Trends cache key embeds explicit dates (not `today 3-m`), so each day gets a fresh window without `refresh`.
- Concurrent tier scoring means scripted-order stubs would race — report tests use the label-routing `_TierStub` instead.
