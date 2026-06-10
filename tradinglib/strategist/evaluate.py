"""Forward evaluation: paper-trade a ticket's levels on the bars after issue.

The ``evaluate_tickets`` follow-up promised by the FA-tournament spec: every
nightly ticket is timestamped before outcomes are known, and this module
scores it afterwards from daily OHLC alone. Stock-level only — the levels are
the strategy's contract; option structures are execution vehicles. Fills are
conservative: stop-style triggers (entry stops, protective stops) that gap
through their level fill at the open (never better than the plan); limit
entries that gap through fill at the gap open, as a real limit order would
(potentially better than the plan). A bar that touches both stop and target
counts as stopped and is flagged ambiguous. On the bar that fills a limit
entry, a target-only touch does not exit — intrabar order is unknowable and
the target rally may have preceded the dip that filled the entry; a stop hit
on that bar still exits. R-multiples use the PLANNED risk |entry - stop| — a
market entry can gap past the stop, which would make fill-based risk degenerate.
"""

from __future__ import annotations

import pandas as pd

ENTRY_WINDOW = 5  # sessions; levels are "tomorrow's numbers" — a stale trigger is a different trade


def _sessions_after(bars: pd.DataFrame, asof: str) -> pd.DataFrame:
    """Bars strictly after the issue date, on a tz-naive index."""
    idx = bars.index
    if getattr(idx, "tz", None) is not None:
        bars = bars.set_axis(idx.tz_convert("UTC").tz_localize(None))
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    return bars.loc[bars.index > pd.Timestamp(asof)]


def _date(ts: object) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


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


def simulate_ticket(
    ticket: dict, bars: pd.DataFrame, *, asof: str, entry_window: int = ENTRY_WINDOW
) -> dict:
    """Paper-trade one ticket's levels on the daily bars after ``asof``. Pure; no I/O."""
    levels = ticket["levels"]
    stance = ticket["stance"]
    sign = 1 if stance == "long" else -1
    entry, stop, target = (float(levels[k]) for k in ("entry", "stop", "target"))
    risk = abs(entry - stop)
    if risk == 0.0:
        raise ValueError(f"degenerate levels: entry == stop == {entry}")

    after = _sessions_after(bars, asof)
    out: dict = {
        "status": "waiting",
        "entry_date": None,
        "entry_fill": None,
        "exit_date": None,
        "exit_fill": None,
        "r": None,
        "pct_move": None,
        "sessions_held": 0,
        "ambiguous_bar": False,
        "closes": [],
    }

    entry_idx: int | None = None
    for i in range(min(entry_window, len(after))):
        fill = _entry_fill(after.iloc[i], entry, levels["entry_type"], stance)
        if fill is not None:
            entry_idx = i
            out["entry_date"] = _date(after.index[i])
            out["entry_fill"] = fill
            break
    if entry_idx is None:
        shown = after.head(entry_window)
        out["closes"] = [[_date(ts), float(row["close"])] for ts, row in shown.iterrows()]
        out["status"] = "expired" if len(after) >= entry_window else "waiting"
        return out

    out["status"] = "open"
    is_limit = levels["entry_type"] == "limit"
    for bar_num, (ts, row) in enumerate(after.iloc[entry_idx:].iterrows()):
        out["sessions_held"] += 1
        fill, status, ambiguous = _exit_fill(row, stop, target, stance)
        if fill is not None:
            # On the limit-entry bar, a target-only hit is suppressed: intrabar
            # order is unknowable and the rally may have preceded the fill dip.
            # A stop hit still exits (price must pass through entry to reach stop).
            if bar_num == 0 and is_limit and status == "target":
                continue
            out["status"] = status
            out["exit_date"] = _date(ts)
            out["exit_fill"] = fill
            out["ambiguous_bar"] = ambiguous
            break

    end = entry_idx + out["sessions_held"]
    out["closes"] = [[_date(ts), float(row["close"])] for ts, row in after.iloc[:end].iterrows()]
    reference = (
        out["exit_fill"] if out["exit_fill"] is not None else float(after.iloc[end - 1]["close"])
    )
    out["r"] = sign * (reference - out["entry_fill"]) / risk
    out["pct_move"] = sign * (reference - out["entry_fill"]) / out["entry_fill"]
    return out
