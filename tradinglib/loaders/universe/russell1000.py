"""Russell 1000 constituent-universe loader (Wikipedia default provider).

Same canonical schema as the S&P 500 loader —
``[ticker, name, sector, sub_industry, cik]`` — but the Wikipedia Russell
1000 components table has no CIK column, so CIKs are joined from the SEC's
``company_tickers.json`` (see :mod:`tradinglib.loaders.universe.cik_map`).
``cik`` is therefore nullable (``Int64``): tickers with no SEC match keep
``<NA>`` and downstream EDGAR stages must skip them.

Cached to ``data/processed/universe/russell1000/<snapshot>.parquet``. On
scrape failure the most recent cached snapshot is returned instead (and the
failure logged); with no cache at all the error propagates.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import cast

import httpx
import pandas as pd

from tradinglib.data.paths import processed_dir
from tradinglib.loaders.edgar_client import EdgarClient
from tradinglib.loaders.universe.cik_map import get_cik_map
from tradinglib.loaders.universe.sp500 import _normalize_ticker

SOURCE = "universe"
_SUBDIR = "russell1000"
_WIKI_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

logger = logging.getLogger(__name__)


def _canonicalize(raw: pd.DataFrame, cik_map: dict[str, int]) -> pd.DataFrame:
    """Normalize the Wikipedia components table into the canonical schema."""
    tickers = raw["Symbol"].astype(str).map(_normalize_ticker)
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "name": raw["Company"].astype(str),
            "sector": raw["GICS Sector"].astype(str),
            "sub_industry": raw["GICS Sub-Industry"].astype(str),
            "cik": pd.array([cik_map.get(t) for t in tickers], dtype="Int64"),
        }
    )
    df = df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    return cast(pd.DataFrame, df[["ticker", "name", "sector", "sub_industry", "cik"]])


def _download(cik_map: dict[str, int]) -> pd.DataFrame:
    """Scrape the components table from Wikipedia."""
    response = httpx.get(
        _WIKI_URL,
        headers={"User-Agent": "trading-models research (van.kyle.00@gmail.com)"},
        timeout=30.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        if "Symbol" in table.columns and "Company" in table.columns:
            return _canonicalize(table, cik_map)
    raise ValueError("no components table found on the Wikipedia page")


def get_russell1000_constituents(
    *, refresh: bool = False, client: EdgarClient | None = None
) -> pd.DataFrame:
    """Return the current Russell 1000 constituent list.

    Snapshot-cached per download date. On scrape failure the most recent
    cached snapshot is returned with a warning; with no cache the error
    propagates (a failed nightly run is visible, a silent empty one is not).
    """
    snapshot = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    cache_dir = processed_dir(SOURCE) / _SUBDIR
    out = cache_dir / f"{snapshot}.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
        df.attrs["snapshot"] = snapshot
        return df
    try:
        df = _download(get_cik_map(refresh=refresh, client=client))
    except Exception:
        cached = sorted(cache_dir.glob("*.parquet"))
        if not cached:
            raise
        logger.warning(
            "Russell 1000 scrape failed; serving the %s snapshot", cached[-1].stem, exc_info=True
        )
        stale = pd.read_parquet(cached[-1])
        stale.attrs["snapshot"] = cached[-1].stem
        return stale
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.attrs["snapshot"] = snapshot
    return df
