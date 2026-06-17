# tradinglib/eval/trades.py
"""Extract discrete trades from a per-bar position series.

A trade opens when the position moves away from flat (0) and closes when it
returns to flat or flips sign. An open position at the end of the series is
closed on the final bar at its price. PnL is the price change over the holding
window times the (signed) position size held; ``duration`` is the number of
bars between entry and exit (``exit_i - entry_i``), not counting the exit bar.

This is the shared source of truth for the trade table and the price-chart
buy/sell markers (which previously lived inline in the Streamlit data view).
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl", "duration"]


def trades_from_position(
    position: pd.Series, prices: pd.Series, *, include_open: bool = True
) -> pd.DataFrame:
    """Return a DataFrame of round-trip trades inferred from ``position``.

    ``position`` and ``prices`` are bar-indexed; ``prices`` is reindexed onto
    ``position``'s index. Each row: entry_time, exit_time, side
    ('long'/'short'), entry_price, exit_price, pnl, duration (bars between
    entry and exit, not counting the exit bar).

    By default a position still open on the final bar is force-closed there and
    counted as a round-trip (the trade-table/chart-marker behavior). Pass
    ``include_open=False`` to drop that synthetic trade — required wherever the
    count is used as a statistic (e.g. the tournament's ``min_trades`` survival
    gate), so an unclosed position cannot pad the trade count.

    Raises ``ValueError`` if ``prices`` has no value for a bar in ``position``'s
    index — a silently NaN-filled price would corrupt the trade PnL.
    """
    prices = prices.reindex(position.index)
    if prices.isna().any():
        missing = prices.index[prices.isna()].tolist()
        raise ValueError(
            f"prices is missing values at {missing[:5]} (and possibly more); "
            "prices and position must share the same bar calendar"
        )
    rows: list[dict] = []

    # Position values come from model signals and are exactly 0.0 / ±1.0 (or
    # integer multiples), so float `== 0.0` comparison is safe here.
    open_size = 0.0
    entry_i = 0
    for i in range(len(position)):
        size = float(position.iloc[i])
        if open_size == 0.0 and size != 0.0:
            open_size, entry_i = size, i
        elif open_size != 0.0 and (size == 0.0 or (size > 0) != (open_size > 0)):
            rows.append(_close(position, prices, entry_i, i, open_size))
            open_size = size
            entry_i = i if size != 0.0 else entry_i

    if open_size != 0.0 and include_open:
        rows.append(_close(position, prices, entry_i, len(position) - 1, open_size))

    return pd.DataFrame(rows, columns=COLUMNS)


def _close(
    position: pd.Series,
    prices: pd.Series,
    entry_i: int,
    exit_i: int,
    size: float,
) -> dict:
    entry_price = float(prices.iloc[entry_i])
    exit_price = float(prices.iloc[exit_i])
    return {
        "entry_time": position.index[entry_i],
        "exit_time": position.index[exit_i],
        "side": "long" if size > 0 else "short",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": (exit_price - entry_price) * (1.0 if size > 0 else -1.0) * abs(size),
        "duration": exit_i - entry_i,
    }
