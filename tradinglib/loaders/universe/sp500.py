"""S&P 500 constituent-universe loader (Wikipedia default provider).

Schema (canonical): a DataFrame with columns
``[ticker, name, sector, sub_industry, cik]``. ``sector``/``sub_industry``
are GICS labels (used for sector-relative valuation) and ``cik`` is the SEC
identifier (the join key for EDGAR). Cached to
``data/processed/universe/sp500/<snapshot>.parquet`` with the same
point-in-time snapshot discipline as the earnings loader.

If the Wikipedia scrape fails, a vendored static CSV snapshot is returned
instead (and intentionally NOT cached, so the next run retries the scrape).
Tickers are normalized to yfinance style (``BRK.B`` -> ``BRK-B``).
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import cast

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir
from tradinglib.provenance import MEMBERSHIP_SURVIVORSHIP, from_reasons

SOURCE = "universe"
_SUBDIR = "sp500"
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_STATIC_CSV = Path(__file__).parent / "sp500_static.csv"

logger = logging.getLogger(__name__)


def _normalize_ticker(symbol: str) -> str:
    """Wikipedia uses dots for share classes; yfinance uses dashes."""
    return symbol.strip().upper().replace(".", "-")


def _canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Wikipedia constituents table into the canonical schema."""
    df = pd.DataFrame(
        {
            "ticker": raw["Symbol"].astype(str).map(_normalize_ticker),
            "name": raw["Security"].astype(str),
            "sector": raw["GICS Sector"].astype(str),
            "sub_industry": raw["GICS Sub-Industry"].astype(str),
            "cik": raw["CIK"].astype("int64"),
        }
    )
    df = df.sort_values("ticker").reset_index(drop=True)
    return cast(pd.DataFrame, df[["ticker", "name", "sector", "sub_industry", "cik"]])


def _download() -> pd.DataFrame:
    """Scrape the constituents table from Wikipedia."""
    response = httpx.get(
        _WIKI_URL,
        headers={"User-Agent": "trading-models research (van.kyle.00@gmail.com)"},
        timeout=30.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        if "Symbol" in table.columns and "CIK" in table.columns:
            return _canonicalize(table)
    raise ValueError("no constituents table found on the Wikipedia page")


def _static_fallback() -> pd.DataFrame:
    df = pd.read_csv(_STATIC_CSV, dtype={"cik": "int64"})
    return cast(pd.DataFrame, df[["ticker", "name", "sector", "sub_industry", "cik"]])


def get_sp500_constituents(*, refresh: bool = False) -> pd.DataFrame:
    """Return the current S&P 500 constituent list.

    Snapshot-cached per download date. On scrape failure the committed static
    CSV is returned with a warning; that fallback is not written to the cache.
    """
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = processed_dir(SOURCE) / _SUBDIR / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        try:
            df = _download()
        except Exception:
            logger.warning("S&P 500 scrape failed; using the vendored static list", exc_info=True)
            df = _static_fallback()
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
    # Membership is today's Wikipedia table back-projected onto any as-of date:
    # survivorship + inclusion look-ahead, unfixable on free data (audit #89).
    df.attrs["provenance"] = from_reasons(MEMBERSHIP_SURVIVORSHIP)
    return df
