"""Forward option-chain snapshot collector (yfinance).

Same canonical schema as the DoltHub loader (``[date, ticker, expiration,
strike, right, bid, ask, iv]``) plus a ``spot`` column, so the backtest can
consume either source once enough forward history accrues. Snapshots are
point-in-time by definition: one parquet per (ticker, snapshot date) at
``data/processed/options/yf_snapshots/<ticker>/<date>.parquet``.

Dates are US Eastern Time (America/New_York) so an evening UTC run does not
stamp tomorrow's date.

Idempotency: an existing file for today is never re-fetched.  Return sentinels:
  -1  already snapshotted today (file exists, skipped)
  -2  fetch failed — no file written; re-run to retry
   0  yfinance returned no expirations in window — no file written; re-run to retry
  >0  rows written

Empty results are never persisted: if yfinance transiently returns no
expirations (or all beyond the cutoff) the run returns 0 and writes nothing, so
the next run retries rather than being locked out by the idempotency guard.

Only expirations within ``MAX_DAYS`` calendar days are collected (enough to
cover earnings straddle tenors). yfinance is mocked in tests and never called
live (repo convention).

``fetch_chain`` adds ``open_interest``/``volume`` in memory only (never
persisted); the snapshot schema remains the canonical nine columns above.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "options"
_SUBDIR = "yf_snapshots"
MAX_DAYS = 45

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv", "spot"]

FETCH_MAX_DAYS = 120  # ticket-playbook window: covers 60-90 DTE directional structures

CHAIN_COLUMNS = [*_COLUMNS, "open_interest", "volume"]


def _optional_numeric(leg: pd.DataFrame, col: str) -> pd.Series | float:
    """Column coerced to numeric, or NaN when yfinance omits it."""
    return pd.to_numeric(leg[col], errors="coerce") if col in leg.columns else float("nan")


def _canonicalize_expiry(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    ticker: str,
    snapshot: str,
    expiration: str,
    spot: float,
) -> pd.DataFrame:
    frames = []
    for right, leg in (("call", calls), ("put", puts)):
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp(snapshot),
                    "ticker": ticker,
                    "expiration": pd.Timestamp(expiration),
                    "strike": leg["strike"].astype(float),
                    "right": right,
                    "bid": pd.to_numeric(leg["bid"], errors="coerce"),
                    "ask": pd.to_numeric(leg["ask"], errors="coerce"),
                    "iv": pd.to_numeric(leg["impliedVolatility"], errors="coerce"),
                    "spot": float(spot),
                    "open_interest": _optional_numeric(leg, "openInterest"),
                    "volume": _optional_numeric(leg, "volume"),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)


def _fetch_frames(ticker: str, *, snapshot: str, cutoff: pd.Timestamp) -> list[pd.DataFrame]:
    """One canonical frame per in-window expiration; raises on yfinance failure."""
    t = yf.Ticker(ticker)
    spot = float(t.fast_info["lastPrice"])
    frames = []
    for exp in t.options:
        if pd.Timestamp(exp) > cutoff:
            continue
        chain = t.option_chain(exp)  # fetch once per expiration
        frames.append(
            _canonicalize_expiry(
                chain.calls,
                chain.puts,
                ticker=ticker,
                snapshot=snapshot,
                expiration=exp,
                spot=spot,
            )
        )
    return frames


def fetch_chain(ticker: str, *, max_days: int = FETCH_MAX_DAYS) -> pd.DataFrame:
    """In-memory near-term chain for the ticket playbook (never persisted).

    Canonical snapshot schema plus ``open_interest``/``volume`` (the playbook's
    liquidity gate needs them). Raises on fetch failure — the scan pipeline
    isolates per-ticker errors; an empty frame means yfinance listed no
    expirations inside the window.
    """
    now_et = pd.Timestamp.now(tz="America/New_York")
    snapshot = now_et.strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(snapshot) + pd.Timedelta(days=max_days)
    frames = _fetch_frames(ticker, snapshot=snapshot, cutoff=cutoff)
    if not frames:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def snapshot_chains(tickers: list[str], *, max_days: int = MAX_DAYS) -> dict[str, int]:
    """Snapshot near-term chains; returns {ticker: rows written}.

    Sentinels: -1 = already snapshotted today, -2 = fetch failed, 0 = no data
    (nothing written — re-run to retry), >0 = rows written.
    """
    now_et = pd.Timestamp.now(tz="America/New_York")
    snapshot = now_et.strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(snapshot) + pd.Timedelta(days=max_days)
    counts: dict[str, int] = {}
    for ticker in tickers:
        out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
        if out.exists():
            counts[ticker] = -1
            continue
        try:
            frames = _fetch_frames(ticker, snapshot=snapshot, cutoff=cutoff)
            if not frames:
                counts[ticker] = 0
                continue
            df = pd.concat(frames, ignore_index=True)[_COLUMNS]
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            counts[ticker] = len(df)
        except Exception:
            counts[ticker] = -2
            continue
    return counts
