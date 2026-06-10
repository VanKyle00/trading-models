"""Ticker -> CIK mapping from the SEC's ``company_tickers.json``.

The Russell 1000 Wikipedia constituents table has no CIK column (unlike the
S&P 500 page), so universes built from it join CIKs through this mapping.
The SEC file is a flat JSON object of ``{cik_str, ticker, title}`` entries;
share-class tickers already use yfinance-style dashes (``BRK-B``). Cached to
``data/processed/universe/cik_map/<snapshot>.parquet`` per download day.
"""

from __future__ import annotations

import pandas as pd

from tradinglib.data.paths import processed_dir
from tradinglib.loaders.edgar_client import EdgarClient

SOURCE = "universe"
_SUBDIR = "cik_map"
_URL = "https://www.sec.gov/files/company_tickers.json"

_default_client: EdgarClient | None = None


def _get_client(client: EdgarClient | None) -> EdgarClient:
    global _default_client
    if client is not None:
        return client
    if _default_client is None:
        _default_client = EdgarClient()
    return _default_client


def get_cik_map(*, refresh: bool = False, client: EdgarClient | None = None) -> dict[str, int]:
    """Return ``{ticker: cik}`` for every SEC-registered ticker.

    Snapshot-cached per day. First occurrence wins on duplicate tickers
    (the SEC file lists larger registrants first).
    """
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
        return dict(zip(df["ticker"], df["cik"], strict=True))

    payload = _get_client(client).get(_URL).json()
    mapping: dict[str, int] = {}
    for entry in payload.values():
        ticker = str(entry["ticker"]).strip().upper()
        if ticker and ticker not in mapping:
            mapping[ticker] = int(entry["cik_str"])

    df = pd.DataFrame({"ticker": list(mapping), "cik": list(mapping.values())})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return mapping
