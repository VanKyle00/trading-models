"""Forward option-chain snapshot collector (yfinance).

Same canonical schema as the DoltHub loader (``[date, ticker, expiration,
strike, right, bid, ask, iv]``) plus a ``spot`` column, so the backtest can
consume either source once enough forward history accrues. Snapshots are
point-in-time by definition: one parquet per (ticker, snapshot date) at
``data/processed/options/yf_snapshots/<ticker>/<date>.parquet``, and an
existing file for today is never re-fetched (idempotent — a snapshot has no
meaningful "refresh"). Only expirations within ``MAX_DAYS`` calendar days are
collected (enough to cover earnings straddle tenors). yfinance is mocked in
tests and never called live (repo convention).
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "options"
_SUBDIR = "yf_snapshots"
MAX_DAYS = 45

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv", "spot"]


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
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([], dtype="object") for c in _COLUMNS})


def snapshot_chains(tickers: list[str], *, max_days: int = MAX_DAYS) -> dict[str, int]:
    """Snapshot near-term chains; returns {ticker: rows written} (-1 = already done today)."""
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    cutoff = pd.Timestamp(snapshot) + pd.Timedelta(days=max_days)
    counts: dict[str, int] = {}
    for ticker in tickers:
        out = processed_dir(SOURCE) / _SUBDIR / ticker / f"{snapshot}.parquet"
        if out.exists():
            counts[ticker] = -1
            continue
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
        df = pd.concat(frames, ignore_index=True) if frames else _empty()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        counts[ticker] = len(df)
    return counts
