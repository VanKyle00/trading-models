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

import pandas as pd


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


def order_flow_imbalance(trades: pd.DataFrame, bar_seconds: int = 60) -> pd.Series:
    """Per-bar order flow imbalance: signed taker volume / total volume.

    Returns a Series indexed by the right-aligned bar timestamps. Range is
    ``[-1, +1]`` — positive means aggressive buyers dominated the bar.
    """
    return aggregate_to_bars(trades, bar_seconds)["ofi"]
