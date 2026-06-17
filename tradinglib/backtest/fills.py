"""Intrabar fill primitives — the single source of truth for how a resting
stop/limit order fills against one daily OHLC bar.

Shared by the forward ledger (:func:`tradinglib.strategist.evaluate.simulate_ticket`)
and the tournament's resting-order scoring (:mod:`tradinglib.backtest.resting_engine`)
so the certified backtest and the issued/scored ticket can never drift apart
(audit A2). Fills are conservative: stop-style triggers that gap through their
level fill at the open (never better than the plan); limit entries that gap
through fill at the gap open (potentially better than the plan, as a real limit
order would). A bar that touches both stop and target counts as stopped and is
flagged ambiguous.
"""

from __future__ import annotations

import pandas as pd


def _entry_fill(row: pd.Series, entry: float, entry_type: str, stance: str) -> float | None:
    """Fill price if this bar triggers the entry, else None."""
    o, hi, lo = float(row["open"]), float(row["high"]), float(row["low"])
    if entry_type == "market":
        return o
    if entry_type == "stop":  # long: buy as price rises to entry; short: sell as it falls
        if stance == "long":
            return max(o, entry) if hi >= entry else None
        return min(o, entry) if lo <= entry else None
    if entry_type == "limit":  # long: buy the dip to entry; short: sell the rip to entry
        if stance == "long":
            return min(o, entry) if lo <= entry else None
        return max(o, entry) if hi >= entry else None
    raise ValueError(f"unknown entry_type {entry_type!r}")


def _exit_fill(
    row: pd.Series, stop: float, target: float, stance: str
) -> tuple[float | None, str, bool]:
    """(fill, status, ambiguous) if this bar exits the trade, else (None, "", False)."""
    o, hi, lo = float(row["open"]), float(row["high"]), float(row["low"])
    if stance == "long":
        stop_hit, target_hit = lo <= stop, hi >= target
        stop_px, target_px = min(o, stop), max(o, target)
    else:
        stop_hit, target_hit = hi >= stop, lo <= target
        stop_px, target_px = max(o, stop), min(o, target)
    if stop_hit:  # worst case wins when both levels are touched intrabar
        return stop_px, "stopped", target_hit
    if target_hit:
        return target_px, "target", False
    return None, "", False
