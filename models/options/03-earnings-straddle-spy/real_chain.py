"""Quote-to-quote real-chain event mechanics for the earnings straddle (Phase 2).

P&L contains NO pricing model: entry buys the ATM call+put at the real ask,
exit sells both at the real bid. A zero (or null) exit bid is a valid
total-loss outcome — only absent contract rows at exit count as missing data.
The k-gate consumes the real implied move ``(call_mid + put_mid) / spot`` via
the unchanged ``signal.py`` functions, so the gate is name-specific for the
first time (Phase 1's synthetic implied move was ~0.075 for every ticker).

Skip reasons are first-class and mutually exclusive; the runner counts and
reports every one (no silent truncation). Deviation from the design spec,
deliberate: ``signal.tradeable_event`` is NOT used because it returns False
when ``expected`` is NaN (no prior earnings history), which would silently
drop forecast-less events from the *unfiltered* baseline; the spread cap is
checked directly and ``signal.passes_filter`` already maps NaN -> gate off.

Timing mirrors ``strategy.py``: the earnings bar is the first bar at/after the
earnings datetime; entry is ``entry_lead`` bars before it, exit is
``exit_offset`` bars after it. Expiry selection is *strictly after* the
earnings datetime: a same-day expiry settles at that day's close, before an
AMC announcement move is realized, so it cannot capture the event.

Fees mirror Phase 1's ``fee_bps=1.0``: each crossing pays fee_bps on its
notional, so ``fees = (entry_cost + exit_value) * 100 * fee_bps / 1e4``.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from tradinglib.options.instruments import CONTRACT_MULTIPLIER

_HERE = Path(__file__).resolve().parent

SKIP_NO_ENTRY_CHAIN = "no_entry_chain"
SKIP_NO_EXPIRY = "no_post_earnings_expiry"
SKIP_NO_QUOTED_ATM = "no_quoted_atm"
SKIP_SPREAD = "spread_over_cap"
SKIP_NO_EXIT_CHAIN = "no_exit_chain"

SKIP_REASONS = [
    SKIP_NO_ENTRY_CHAIN,
    SKIP_NO_EXPIRY,
    SKIP_NO_QUOTED_ATM,
    SKIP_SPREAD,
    SKIP_NO_EXIT_CHAIN,
]

_MAX_LOOKBACK = 20  # prior_moves stored per event are trimmed to this for sweeps


def _load_signal():
    spec = importlib.util.spec_from_file_location("es_signal", _HERE / "signal.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIG = _load_signal()


def pick_expiry(chain: pd.DataFrame, earnings_datetime: pd.Timestamp) -> pd.Timestamp | None:
    """Nearest available expiration STRICTLY after the earnings datetime."""
    if chain.empty:
        return None
    after = sorted(e for e in set(chain["expiration"]) if e > earnings_datetime)
    return after[0] if after else None


def pick_atm_strike(chain: pd.DataFrame, expiry: pd.Timestamp, spot: float) -> float | None:
    """Nearest strike to spot with BOTH legs quoted (bid > 0 and ask > 0 on call AND put)."""
    sub = chain[chain["expiration"] == expiry]
    quoted = sub[(sub["bid"] > 0) & (sub["ask"] > 0)]
    if quoted.empty:
        return None
    rights_by_strike = quoted.groupby("strike")["right"].agg(set)
    both = [s for s, rights in rights_by_strike.items() if rights >= {"call", "put"}]
    if not both:
        return None
    return float(min(both, key=lambda s: abs(s - spot)))


def straddle_quotes(
    chain: pd.DataFrame, expiry: pd.Timestamp, strike: float
) -> dict[str, float] | None:
    """{call_bid, call_ask, put_bid, put_ask} for one contract pair, or None if a leg is absent."""
    sub = chain[(chain["expiration"] == expiry) & (chain["strike"] == strike)]
    out: dict[str, float] = {}
    for right in ("call", "put"):
        leg = sub[sub["right"] == right]
        if leg.empty:
            return None
        out[f"{right}_bid"] = float(leg["bid"].iloc[0])
        out[f"{right}_ask"] = float(leg["ask"].iloc[0])
    return out


def past_abs_moves(close: pd.Series, earnings_datetimes: pd.Series) -> list[float]:
    """Chronological abs earnings-day returns — the same definition as
    ``signal.expected_move`` (close at the last bar <= the event to the next
    bar), returned as a list so lookback sweeps can re-slice it."""
    closes = close.sort_index()
    cidx = closes.index
    if getattr(cidx, "tz", None) is not None:
        cidx = cidx.tz_localize(None)
        closes = pd.Series(closes.to_numpy(), index=cidx)
    eds = pd.to_datetime(earnings_datetimes, utc=True).dt.tz_localize(None).sort_values()

    moves: list[float] = []
    for ed in eds:
        prior = closes.index[closes.index <= ed]
        if len(prior) == 0:
            continue
        pos = closes.index.get_loc(prior[-1])
        if pos + 1 >= len(closes):
            continue
        prev = float(closes.iloc[pos])
        if prev <= 0.0:
            continue
        ret = float(closes.iloc[pos + 1]) / prev - 1.0
        if not math.isfinite(ret):
            continue
        moves.append(abs(ret))
    return moves


def _bid_or_zero(x: float) -> float:
    """A null/negative bid means no resting bid — worth zero to close into."""
    return 0.0 if (x is None or math.isnan(x) or x < 0) else x


def run_event(
    *,
    ticker: str,
    close: pd.Series,
    earnings_datetime: pd.Timestamp,
    prior_moves: list[float],
    load_chain: Callable[..., pd.DataFrame],
    entry_lead: int = 3,
    exit_offset: int = 1,
    k: float = 1.2,
    lookback: int = 8,
    max_spread_frac: float = 0.20,
    fee_bps: float = 1.0,
) -> dict[str, Any]:
    """One earnings event, quote-to-quote.

    Returns ``{"skip_reason": <reason>}`` when the chain cannot trade, else a
    full trade record (the gate is evaluated but pnl is always present, so the
    unfiltered branch can use every tradeable event). Raises ValueError when
    the event window does not fit the bar series (the runner's concern).
    ``close`` must be tz-naive; ``prior_moves`` are abs moves of PRIOR events
    only (leakage discipline is the caller's contract, as in Phase 1).
    """
    bars = close.index
    after = bars >= earnings_datetime
    if not after.any():
        raise ValueError(f"event window out of range: earnings {earnings_datetime} beyond bars")
    e_idx = int(after.argmax())
    entry_idx, exit_idx = e_idx - entry_lead, e_idx + exit_offset
    if entry_idx < 0 or exit_idx >= len(bars):
        raise ValueError(
            f"event window out of range: need {entry_lead} bars before and "
            f"{exit_offset} after the earnings bar"
        )
    entry_date, exit_date = bars[entry_idx], bars[exit_idx]
    spot = float(close.iloc[entry_idx])

    entry_chain = load_chain(ticker, entry_date)
    if entry_chain.empty:
        return {"skip_reason": SKIP_NO_ENTRY_CHAIN}
    expiry = pick_expiry(entry_chain, earnings_datetime)
    if expiry is None:
        return {"skip_reason": SKIP_NO_EXPIRY}
    strike = pick_atm_strike(entry_chain, expiry, spot)
    if strike is None:
        return {"skip_reason": SKIP_NO_QUOTED_ATM}
    quotes = straddle_quotes(entry_chain, expiry, strike)
    if quotes is None:
        return {"skip_reason": SKIP_NO_QUOTED_ATM}

    entry_cost = quotes["call_ask"] + quotes["put_ask"]
    entry_bid = quotes["call_bid"] + quotes["put_bid"]
    mid = (entry_cost + entry_bid) / 2.0
    spread_frac = (entry_cost - entry_bid) / mid if mid > 0 else math.inf
    if spread_frac > max_spread_frac:
        return {"skip_reason": SKIP_SPREAD}

    implied = _SIG.implied_move(mid, spot)
    trimmed = [abs(m) for m in prior_moves][-_MAX_LOOKBACK:]
    window = trimmed[-lookback:]
    expected = float(pd.Series(window).mean()) if window else float("nan")

    exit_chain = load_chain(ticker, exit_date)
    exit_quotes = None if exit_chain.empty else straddle_quotes(exit_chain, expiry, strike)
    if exit_quotes is None:
        return {"skip_reason": SKIP_NO_EXIT_CHAIN}
    exit_value = _bid_or_zero(exit_quotes["call_bid"]) + _bid_or_zero(exit_quotes["put_bid"])

    fees = (entry_cost + exit_value) * CONTRACT_MULTIPLIER * (fee_bps / 1e4)
    pnl = (exit_value - entry_cost) * CONTRACT_MULTIPLIER - fees

    return {
        "ticker": ticker,
        "date": str(pd.Timestamp(earnings_datetime).date()),
        "entry_date": str(pd.Timestamp(entry_date).date()),
        "exit_date": str(pd.Timestamp(exit_date).date()),
        "expiry": str(pd.Timestamp(expiry).date()),
        "strike": float(strike),
        "spot": spot,
        "entry_cost": entry_cost,
        "exit_value": exit_value,
        "spread_frac": spread_frac,
        "implied_move": implied,
        "expected_move": expected,
        "prior_moves": trimmed,
        "gate_fired": bool(_SIG.passes_filter(expected, implied, k)),
        "pnl": pnl,
    }


def gate_pnls(events: list[dict], *, k: float, lookback: int) -> list[float]:
    """Recompute gate membership from stored rows (pnl is k/lookback-independent)."""
    fired: list[float] = []
    for ev in events:
        window = ev["prior_moves"][-lookback:]
        expected = float(pd.Series(window).mean()) if window else float("nan")
        if _SIG.passes_filter(expected, ev["implied_move"], k):
            fired.append(float(ev["pnl"]))
    return fired
