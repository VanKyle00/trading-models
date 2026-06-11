"""Stocktwits symbol-stream loader (Tier 3 viral proxy).

Schema (canonical): ``[ticker, created, body, sentiment, username, url]``,
UTC-aware, newest first, capped at ``max_items``. ``sentiment`` is the
user-tagged label ("Bullish"/"Bearish") or ``None`` — free ground truth the
mechanical bull/bear ratio is computed from. Keyless public endpoint
(~200 req/hr/IP). Any fetch failure (unknown symbol, rate limit, network) → empty.
Snapshot-cached to
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
            "created": pd.Series([], dtype="datetime64[ms, UTC]"),
            "body": pd.Series([], dtype="object"),
            "sentiment": pd.Series([], dtype="object"),
            "username": pd.Series([], dtype="object"),
            "url": pd.Series([], dtype="object"),
        }
    )


def _normalize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # Pandas 3 + Arrow backend coerces None -> float NaN (in construction AND on
    # parquet read). Rebuild as object dtype so None survives; float NaN would be
    # truthy, breaking `if r.sentiment:` checks in downstream itertuples() loops.
    values = [None if pd.isna(v) else v for v in df["sentiment"]]
    df["sentiment"] = pd.Series(values, dtype=object, index=df.index)
    return df


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
    # ms (not ns) so the dtype survives the parquet round-trip and cached == fresh
    df["created"] = pd.to_datetime(df["created"], utc=True).astype("datetime64[ms, UTC]")
    df = _normalize_sentiment(df)
    return df.sort_values("created", ascending=False).reset_index(drop=True)


def get_stocktwits(ticker: str, *, max_items: int = 30, refresh: bool = False) -> pd.DataFrame:
    """Recent Stocktwits messages for one ticker, newest first."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = _normalize_sentiment(pd.read_parquet(out))
    else:
        df = _download(ticker)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df.head(max_items).reset_index(drop=True)
