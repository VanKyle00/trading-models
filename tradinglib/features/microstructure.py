"""Microstructure features computed from tick-level trade data.

These functions consume a canonicalized trades DataFrame as produced by
``tradinglib.loaders.crypto.binance_trades.load_trades`` — indexed by
``timestamp`` with columns ``price``, ``qty``, ``is_buyer_maker`` — and
return either bar-aggregated DataFrames or per-bar feature Series.

Convention for trade direction:

- ``is_buyer_maker == False`` → aggressive **buy** (taker lifted the ask).
  We assign ``sign = +1``.
- ``is_buyer_maker == True``  → aggressive **sell** (taker hit the bid).
  We assign ``sign = -1``.

This is the standard "taker side" labelling used in microstructure
research: the side of the trade is the side of the *aggressor*.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_SECONDS_PER_DAY = 86_400

_BAR_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "buy_volume",
    "sell_volume",
    "signed_volume",
    "n_trades",
    "ofi",
]


def aggregate_to_bars(trades: pd.DataFrame, bar_seconds: int = 60) -> pd.DataFrame:
    """Aggregate tick trades into time-based bars with OHLCV + microstructure features.

    Parameters
    ----------
    trades:
        DataFrame indexed by UTC timestamp with columns
        ``price``, ``qty``, ``is_buyer_maker``.
    bar_seconds:
        Bar width in seconds.

    Returns
    -------
    DataFrame indexed by **right-aligned** bar timestamps with columns:
    ``open, high, low, close, volume, buy_volume, sell_volume, signed_volume,
    n_trades, ofi``.

    The right-aligned label means the bar ending at time ``t`` contains all
    trades with timestamp in ``(t - bar_seconds, t]``.
    """
    if not {"price", "qty", "is_buyer_maker"}.issubset(trades.columns):
        raise ValueError("trades must have columns price, qty, is_buyer_maker")

    qty = trades["qty"]
    is_maker = trades["is_buyer_maker"]

    work = pd.DataFrame(
        {
            "price": trades["price"],
            "qty": qty,
            "buy_qty": qty.where(~is_maker, 0.0),
            "sell_qty": qty.where(is_maker, 0.0),
            "signed_qty": qty.where(~is_maker, -qty),
        },
        index=trades.index,
    )

    rule = f"{bar_seconds}s"
    grouped = work.resample(rule, label="right", closed="right")

    bars = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["qty"].sum(),
            "buy_volume": grouped["buy_qty"].sum(),
            "sell_volume": grouped["sell_qty"].sum(),
            "signed_volume": grouped["signed_qty"].sum(),
            "n_trades": grouped["price"].count().astype("int64"),
        }
    )
    bars = bars.dropna(subset=["close"]).copy()
    bars["ofi"] = bars["signed_volume"] / bars["volume"].where(bars["volume"] > 0)
    return bars


def aggregate_daily_chunks(
    daily_trades: Iterable[pd.DataFrame], bar_seconds: int = 60
) -> pd.DataFrame:
    """Aggregate trades to bars one chunk at a time, bounding peak memory.

    Equivalent to ``aggregate_to_bars(pd.concat(list(daily_trades)),
    bar_seconds)`` but never holds more than a single chunk of raw trades in
    memory: each chunk is aggregated to bars (a tiny DataFrame) and the raw
    trades are released before the next chunk is loaded. This is what lets the
    microstructure model run on memory-constrained hosts where concatenating
    every day's millions of ticks would exhaust RAM.

    ``daily_trades`` should be a *lazy* iterable (e.g. a generator yielding one
    day at a time) for the memory bound to hold. Each element must be a
    canonical trades DataFrame as consumed by :func:`aggregate_to_bars`.

    Chunks are expected to be split on calendar-day boundaries. Because bars are
    right-closed, the bar at midnight ``(23:59:..., 00:00:00]`` can straddle two
    adjacent chunks; such boundary bars are re-merged so the result is identical
    to aggregating the full concatenation. This requires ``bar_seconds`` to
    divide evenly into a day so per-chunk bar grids align at midnight.
    """
    if _SECONDS_PER_DAY % bar_seconds != 0:
        raise ValueError(
            f"bar_seconds ({bar_seconds}) must divide evenly into a day "
            f"({_SECONDS_PER_DAY}s) for per-day streaming aggregation"
        )

    per_chunk: list[pd.DataFrame] = []
    for chunk in daily_trades:
        per_chunk.append(aggregate_to_bars(chunk, bar_seconds))
        # `chunk` is rebound on the next iteration; the heavy raw-trade frame is
        # released while only the small bar frame is retained.
    if not per_chunk:
        raise ValueError("no trade chunks to aggregate")

    bars = pd.concat(per_chunk)
    if bars.index.has_duplicates:
        bars = _combine_boundary_bars(bars)
    return bars.sort_index()


def _combine_boundary_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Re-merge bars that share a timestamp because they straddled a chunk edge.

    The OHLCV/flow aggregation is associative, so merging two partial bars for
    the same minute reproduces the single bar the full aggregation would emit:
    open from the earliest partial, close from the latest, high/low as the
    extremes, and volumes/counts summed. Rows with a unique timestamp pass
    through unchanged (single-row groups), so this is safe to run over the whole
    frame.
    """
    grouped = bars.groupby(level=0, sort=True)
    combined = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "buy_volume": grouped["buy_volume"].sum(),
            "sell_volume": grouped["sell_volume"].sum(),
            "signed_volume": grouped["signed_volume"].sum(),
            "n_trades": grouped["n_trades"].sum(),
        }
    )
    combined["ofi"] = combined["signed_volume"] / combined["volume"].where(combined["volume"] > 0)
    return combined[_BAR_COLUMNS]


def order_flow_imbalance(trades: pd.DataFrame, bar_seconds: int = 60) -> pd.Series:
    """Per-bar order flow imbalance: signed taker volume / total volume.

    Returns a Series indexed by the right-aligned bar timestamps. Range is
    ``[-1, +1]`` — positive means aggressive buyers dominated the bar.
    """
    return aggregate_to_bars(trades, bar_seconds)["ofi"]
