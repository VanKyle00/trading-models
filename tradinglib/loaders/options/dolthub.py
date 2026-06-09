"""DoltHub historical option-chain loader (post-no-preference/options).

Free SQL API over the community options database. Upstream schema:
``option_chain(date, act_symbol, expiration, strike, call_put, bid, ask, vol,
delta, gamma, theta, vega, rho)`` with PK ``(date, act_symbol, expiration,
strike, call_put)``. QUERY DISCIPLINE: every query must filter on exact
``date`` AND ``act_symbol`` (the PK prefix) — anything else scans a ~1e9-row
table and times out ("context deadline exceeded", observed 2026-06-09). The
dataset carries only 3-4 expirations per (date, symbol) — select weeklies and
monthlies roughly 2-7 weeks out — so consumers pick the nearest *available*
expiry, not an arbitrary listed one.

Canonical schema: ``[date, ticker, expiration, strike, right, bid, ask, iv]``
with ``right in {"call", "put"}``; decimals arrive as JSON strings and are
coerced to float. Cached to
``data/processed/options/dolthub/<ticker>/<date>.parquet``; an empty API
result caches an empty frame so the miss is remembered. httpx is mocked in
tests and never called live (repo convention).
"""

from __future__ import annotations

import time
import warnings
from datetime import date, datetime

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir

SOURCE = "options"
_SUBDIR = "dolthub"
_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
_OK_STATUSES = ("Success", "RowLimit")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "ticker": pd.Series([], dtype="object"),
            "expiration": pd.Series([], dtype="datetime64[ns]"),
            "strike": pd.Series([], dtype="float64"),
            "right": pd.Series([], dtype="object"),
            "bid": pd.Series([], dtype="float64"),
            "ask": pd.Series([], dtype="float64"),
            "iv": pd.Series([], dtype="float64"),
        }
    )


def _canonicalize(rows: list[dict], ticker: str) -> pd.DataFrame:
    """API JSON rows -> canonical frame (strings coerced, Call/Put lowered)."""
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "ticker": ticker,
            "expiration": pd.to_datetime(df["expiration"]),
            "strike": df["strike"].astype(float),
            "right": df["call_put"].str.lower(),
            "bid": pd.to_numeric(df["bid"], errors="coerce"),
            "ask": pd.to_numeric(df["ask"], errors="coerce"),
            "iv": pd.to_numeric(df["vol"], errors="coerce"),
        }
    )
    return out.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)


def _query(sql: str) -> list[dict]:
    """One SQL query against the DoltHub API; one retry on transient failure.

    Retries cover httpx errors, non-JSON 200 bodies (JSONDecodeError/ValueError),
    and query execution errors.
    """
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = httpx.get(_API, params={"q": sql}, timeout=60.0)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("query_execution_status") not in _OK_STATUSES:
                raise RuntimeError(
                    f"DoltHub query failed: {payload.get('query_execution_message', 'unknown')}"
                )
            if payload.get("query_execution_status") == "RowLimit":
                warnings.warn(
                    f"DoltHub RowLimit hit for query (chain may be truncated): {sql}",
                    stacklevel=2,
                )
            return payload.get("rows", [])
        except (httpx.HTTPError, RuntimeError, ValueError) as err:
            last_err = err
            if attempt == 0:
                time.sleep(1.0)
    raise RuntimeError(f"DoltHub query failed after retry: {last_err}") from last_err


def load_chain(ticker: str, when: str | date | datetime, *, refresh: bool = False) -> pd.DataFrame:
    """Full available chain for one (ticker, trading date), cache-first."""
    d = pd.Timestamp(when).strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{d}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    sql = (
        "SELECT `date`, expiration, strike, call_put, bid, ask, vol "
        f"FROM option_chain WHERE `date`='{d}' AND act_symbol='{ticker}'"
    )
    df = _canonicalize(_query(sql), ticker)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    time.sleep(0.5)  # politeness between live API calls
    return df
