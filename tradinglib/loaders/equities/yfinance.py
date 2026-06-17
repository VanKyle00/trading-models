"""yfinance loader — daily OHLCV bars for US equities, ETFs, indices.

yfinance is unofficial and rate-limited but free and key-less, which makes
it the right choice for portfolio prototyping. For higher-quality or
intraday-level data, prefer Polygon or Alpaca (loaders to come).

Canonical schema written to parquet:
    timestamp: datetime64[ns, UTC]   (index)
    open, high, low, close: float64
    volume: int64
    symbol: str
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pandas as pd
import yfinance as yf

from tradinglib.data.paths import processed_dir

SOURCE = "yfinance"
_CACHE_FILENAME = "daily.parquet"


def load_daily(
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV bars for one symbol, caching to parquet on first run.

    Parameters
    ----------
    symbol:
        Ticker as understood by yfinance (e.g. ``"SPY"``, ``"AAPL"``, ``"BTC-USD"``).
    start, end:
        Optional date filters applied after caching. ``None`` means
        unbounded — yfinance returns the full available history on a fresh
        download.
    refresh:
        If ``True``, redownload and overwrite the cached parquet.
    """
    out = processed_dir(SOURCE) / symbol / _CACHE_FILENAME
    if out.exists() and not refresh:
        df = pd.read_parquet(out)
    else:
        df = _download_daily(symbol)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

    # yfinance occasionally emits a partial trailing row (volume populated,
    # prices NaN); a bar without prices is unusable downstream, and filtering
    # on the read path also heals caches written before this guard existed.
    df = df.dropna(subset=["open", "high", "low", "close"])

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def corp_actions_since(symbol: str, start: str, end: str) -> bool:
    """True if a split or dividend ex-date falls in ``(start, end]``.

    Used by the forward ledger to flag a ticket whose frozen price levels span a
    corporate action: ``auto_adjust`` re-scales the whole price history, so such a
    ticket's frozen-level scoring is unreliable. This is a live network call to
    yfinance's corporate-actions feed; the caller treats any exception as
    "unknown" and proceeds unflagged.
    """
    actions = yf.Ticker(symbol).actions
    if actions is None or actions.empty:
        return False
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    for ts in actions.index:
        ex = pd.Timestamp(ts)
        if ex.tzinfo is not None:
            ex = ex.tz_localize(None)
        if lo < ex <= hi:
            return True
    return False


def _download_daily(symbol: str) -> pd.DataFrame:
    """Hit yfinance and return a canonicalized DataFrame.

    Two fetches: ``auto_adjust=True`` for the canonical (back-adjusted) OHLCV the
    indicators use, and ``auto_adjust=False`` for the UNADJUSTED OHLC the forward
    ledger scores frozen dollar levels against (audit #88). The adjusted columns
    keep coming from the auto_adjust=True fetch verbatim — yfinance applies its
    repair passes before the adjust ratio, so reconstructing adjusted-from-raw is
    NOT byte-identical, and the indicators must stay byte-identical.
    """
    raw = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r}")
    df = _canonicalize(raw, symbol)
    try:
        unadj = yf.download(
            symbol, period="max", interval="1d", auto_adjust=False, actions=False, progress=False
        )
    except Exception:
        unadj = None  # unadjusted is best-effort; the ledger falls back to the A5b path
    return _attach_unadjusted(df, unadj)


def _attach_unadjusted(df: pd.DataFrame, unadj: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``unadj_open/high/low/close`` (float64) from an auto_adjust=False frame,
    aligned to ``df``'s index. All-or-nothing: if the raw OHLC is unavailable the
    columns are simply absent, so the ledger degrades to the A5b corp-action path.
    """
    if unadj is None or unadj.empty:
        return df
    raw = unadj.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    cols = ["open", "high", "low", "close"]
    if not set(cols).issubset(raw.columns):
        return df
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, utc=True), name="timestamp")
    aligned = raw[cols].astype("float64").reindex(df.index)
    for col in cols:
        df[f"unadj_{col}"] = aligned[col]
    return df


def _canonicalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance's output into the canonical schema.

    yfinance returns MultiIndex columns (outer = OHLCV name, inner = ticker)
    even for a single symbol — flatten that. The index is a DatetimeIndex
    with no name in newer yfinance versions; force it to ``timestamp`` and
    make it timezone-aware UTC.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="timestamp")
    df["symbol"] = symbol
    df["volume"] = df["volume"].astype("int64")
    return cast(pd.DataFrame, df[["open", "high", "low", "close", "volume", "symbol"]])
