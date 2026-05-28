"""Google Trends loader — search interest as an attention / sentiment signal.

Uses pytrends (a community wrapper around Google's public Trends API). The
returned values are *relative*: scaled 0-100 within the query window. That
means the series is not directly comparable across windows — query once
over the full window of interest and cache the result.

Resolution depends on the query window:
- < ~9 months → daily
- ~9 months to 5 years → weekly
- > 5 years → monthly

Caveats:
- pytrends is community-maintained and occasionally rate-limited.
- The last row may be flagged ``isPartial`` (the current period isn't done
  yet). The loader filters those out.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from pytrends.request import TrendReq

from tradinglib.data.paths import processed_dir

SOURCE = "google_trends"


def load_interest(
    query: str,
    timeframe: str = "2021-01-01 2024-12-31",
    geo: str = "",
    *,
    refresh: bool = False,
) -> pd.Series:
    """Return the search-interest series for a single query, cached to parquet.

    Parameters
    ----------
    query:
        The keyword to query (e.g. ``"bitcoin"``).
    timeframe:
        pytrends timeframe string ``"YYYY-MM-DD YYYY-MM-DD"``.
    geo:
        ISO country code; empty string = global.
    refresh:
        If True, redownload and overwrite the cache.
    """
    cache_key = f"{query}__{timeframe.replace(' ', '_')}__{geo or 'global'}"
    out = processed_dir(SOURCE) / cache_key / "interest.parquet"
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download(query, timeframe, geo)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return cast(pd.Series, df[query])


def _download(query: str, timeframe: str, geo: str) -> pd.DataFrame:
    """Hit the Trends API and canonicalize the response."""
    pt = TrendReq(hl="en-US", tz=0)
    pt.build_payload(kw_list=[query], timeframe=timeframe, geo=geo)
    df = pt.interest_over_time()
    if df.empty:
        raise ValueError(
            f"Google Trends returned no data for query={query!r}, timeframe={timeframe!r}"
        )
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="timestamp")
    if "isPartial" in df.columns:
        df = df.loc[~df["isPartial"]].drop(columns=["isPartial"])
    return df
