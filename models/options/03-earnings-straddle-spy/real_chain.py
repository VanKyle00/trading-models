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
``exit_offset`` bars after it. Expiry selection is *strictly after* the later
of the earnings datetime and the exit bar: the contract must capture the event
AND still exist at exit (e.g. earnings Thursday AMC with a Friday weekly expiry
and a Monday exit — the Friday contract lapses before exit, so the next expiry
is selected instead).

Intersection expiry selection: the DoltHub dataset lists only ~3 tenor-anchored
expirations per (date, symbol) (~2w/~4w/~7w out; the front week is never
listed), and that visibility window ROLLS day to day — so the expiry nearest
the cutoff at entry frequently is not listed on the exit-date chain even though
the contract genuinely traded continuously. ``run_event`` therefore restricts
candidates to expirations present in BOTH the entry chain and the exit chain
before applying ``pick_expiry``. This is a data-availability workaround for
MARKING the position at both dates, not price lookahead: it conditions on which
rows the dataset happens to publish, never on quote values. Side effect, stated
honestly: the tenor actually held shifts much longer than the nominal spec
(~2 weeks) — empirically 16-64 calendar days, median 53 (~7.5 weeks), because
the shared expiration is usually the far monthly slot rather than the rolling
front slot.

Split handling: DoltHub strikes/quotes are contemporaneous (never
retro-adjusted) while the yfinance closes feeding ``run_event`` are fully
split-adjusted, so ``split_factor`` (cumulative ratio of all splits AFTER the
entry date) rescales the spot back to contemporaneous dollars for strike
selection and the implied move. A ``strike_grid_mismatch`` guard skips any
event whose nearest quoted strike is >15% from the (rescaled) spot, so a
wrong or missing factor can never produce garbage deep-ITM trades.

Coverage snapping: before ~Oct 2024 the DoltHub dataset carries chains only on
Mon/Wed/Fri (Tue/Thu have zero rows table-wide), so loading chains at exactly
the target offsets would structurally select Wed-AMC/Thu-BMO reporters in that
era. ``run_event`` therefore snaps each leg to the nearest COVERED (non-empty)
chain date within a bounded window: entry tries the target first, then
alternates nearer/farther (``entry_snap`` bars each way, never closer than 1
bar before the earnings bar); exit only moves forward (up to ``exit_snap``
bars, never before the earnings bar). The offsets actually used are recorded
per event as ``entry_lead_used``/``exit_offset_used``. If every candidate
chain is empty the event skips with ``no_entry_chain``/``no_exit_chain``.

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

SKIP_NO_ENTRY_CHAIN = "no_entry_chain"  # all entry candidate chains empty
SKIP_NO_EXIT_CHAIN = "no_exit_chain"  # all exit candidate chains empty
SKIP_NO_EXPIRY = "no_post_earnings_expiry"  # no entry-AND-exit expiry strictly after cutoff
SKIP_NO_QUOTED_ATM = "no_quoted_atm"
SKIP_SPREAD = "spread_over_cap"
SKIP_STRIKE_GRID_MISMATCH = "strike_grid_mismatch"  # nearest quoted strike >15% from spot
SKIP_CONTRACT_MISSING_AT_EXIT = "contract_missing_at_exit"  # expiry survived, strike row absent

SKIP_REASONS = [
    SKIP_NO_ENTRY_CHAIN,
    SKIP_NO_EXIT_CHAIN,
    SKIP_NO_EXPIRY,
    SKIP_NO_QUOTED_ATM,
    SKIP_SPREAD,
    SKIP_STRIKE_GRID_MISMATCH,
    SKIP_CONTRACT_MISSING_AT_EXIT,
]

_MAX_LOOKBACK = 20  # prior_moves stored per event are trimmed to this for sweeps


def _load_signal():
    spec = importlib.util.spec_from_file_location("es_signal", _HERE / "signal.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIG = _load_signal()


def pick_expiry(chain: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Timestamp | None:
    """Nearest available expiration STRICTLY after ``cutoff`` (the later of the
    earnings datetime and the exit bar — the contract must capture the event AND
    still exist at exit)."""
    if chain.empty:
        return None
    after = sorted(e for e in set(chain["expiration"]) if e > cutoff)
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
    entry_snap: int = 2,
    exit_snap: int = 3,
    k: float = 1.2,
    lookback: int = 8,
    max_spread_frac: float = 0.20,
    fee_bps: float = 1.0,
    split_factor: float = 1.0,
) -> dict[str, Any]:
    """One earnings event, quote-to-quote.

    Returns ``{"skip_reason": <reason>}`` when the chain cannot trade, else a
    full trade record (the gate is evaluated but pnl is always present, so the
    unfiltered branch can use every tradeable event). Raises ValueError when
    the event window does not fit the bar series (the runner's concern) — the
    fit check uses the TARGET offsets; snap candidates falling outside the bar
    series are simply dropped. Entry/exit snap to the nearest covered chain
    date within ``entry_snap``/``exit_snap`` bars of the targets (see module
    docstring); ``entry_lead_used``/``exit_offset_used`` record the actual
    offsets. ``close`` must be tz-naive; ``prior_moves`` are abs moves of
    PRIOR events only (leakage discipline is the caller's contract, as in
    Phase 1).

    Expiry candidates are the INTERSECTION of entry-chain and exit-chain
    expirations: the contract traded continuously in reality, but the dataset's
    rolling ~3-expiry visibility window must list it at BOTH marking dates —
    a data-availability workaround, not price lookahead (see module docstring;
    held tenors shift longer, ~3-4w vs the nominal ~2w, as a result).

    ``split_factor`` is the cumulative ratio of all splits AFTER the entry date
    (e.g. 20.0 for AMZN before 2022-06-06): ``spot = adjusted_close *
    split_factor`` puts strike selection, ``implied_move``, and the recorded
    ``spot`` in the chain's contemporaneous dollars. Everything downstream
    (quotes, P&L) is already contemporaneous, and ``expected_move`` /
    ``past_abs_moves`` are return-based (split-invariant), so nothing else
    needs rescaling.
    """
    if entry_lead < 1:
        raise ValueError(f"entry_lead must be >= 1 (entry precedes earnings), got {entry_lead}")
    bars = close.index
    after = bars >= earnings_datetime
    if not after.any():
        raise ValueError(f"event window out of range: earnings {earnings_datetime} beyond bars")
    e_idx = int(after.argmax())
    if e_idx - entry_lead < 0 or e_idx + exit_offset >= len(bars):
        raise ValueError(
            f"event window out of range: need {entry_lead} bars before and "
            f"{exit_offset} after the earnings bar"
        )

    # Entry: target offset first, then alternating nearer/farther, bounded to
    # >= 1 bar before the earnings bar and <= entry_lead + entry_snap.
    entry_offsets = [entry_lead]
    for step in range(1, entry_snap + 1):
        entry_offsets += [entry_lead - step, entry_lead + step]
    entry_offsets = [o for o in entry_offsets if o >= 1 and e_idx - o >= 0]
    entry_chain, entry_lead_used = None, entry_lead
    for off in entry_offsets:
        cand = load_chain(ticker, bars[e_idx - off])
        if not cand.empty:
            entry_chain, entry_lead_used = cand, off
            break
    if entry_chain is None:
        return {"skip_reason": SKIP_NO_ENTRY_CHAIN}
    entry_idx = e_idx - entry_lead_used
    entry_date = bars[entry_idx]
    # Contemporaneous spot: yfinance closes are split-adjusted, chain strikes
    # are not — rescale by the cumulative post-entry split factor.
    spot = float(close.iloc[entry_idx]) * split_factor

    # Exit: forward only (must stay strictly after the earnings bar). Resolved
    # before expiry selection because the expiry cutoff must use the SNAPPED
    # exit date; exit quotes are evaluated later from this same chain.
    exit_chain, exit_offset_used = None, exit_offset
    for off in range(exit_offset, exit_offset + exit_snap + 1):
        if e_idx + off >= len(bars):
            break
        cand = load_chain(ticker, bars[e_idx + off])
        if not cand.empty:
            exit_chain, exit_offset_used = cand, off
            break
    if exit_chain is None:
        return {"skip_reason": SKIP_NO_EXIT_CHAIN}
    exit_date = bars[e_idx + exit_offset_used]

    # Intersection: only expirations visible on BOTH marking dates are
    # holdable through exit under the dataset's rolling expiry window
    # (data-availability constraint, not lookahead — see docstrings).
    candidates = entry_chain[entry_chain["expiration"].isin(exit_chain["expiration"].unique())]
    expiry = pick_expiry(candidates, max(earnings_datetime, pd.Timestamp(exit_date)))
    if expiry is None:
        return {"skip_reason": SKIP_NO_EXPIRY}
    strike = pick_atm_strike(entry_chain, expiry, spot)
    if strike is None:
        return {"skip_reason": SKIP_NO_QUOTED_ATM}
    if abs(strike - spot) / spot > 0.15:
        # Strike grid nowhere near spot: a wrong/missing split factor would
        # otherwise buy a garbage deep-ITM "straddle".
        return {"skip_reason": SKIP_STRIKE_GRID_MISMATCH}
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

    exit_quotes = straddle_quotes(exit_chain, expiry, strike)
    if exit_quotes is None:
        # Expiry survived in both chains but the dataset's ~27-strike band is
        # re-sampled daily and can shift grid phase, dropping the strike row.
        return {"skip_reason": SKIP_CONTRACT_MISSING_AT_EXIT}
    exit_value = _bid_or_zero(exit_quotes["call_bid"]) + _bid_or_zero(exit_quotes["put_bid"])

    fees = (entry_cost + exit_value) * CONTRACT_MULTIPLIER * (fee_bps / 1e4)
    pnl = (exit_value - entry_cost) * CONTRACT_MULTIPLIER - fees

    return {
        "ticker": ticker,
        "date": str(pd.Timestamp(earnings_datetime).date()),
        "entry_date": str(pd.Timestamp(entry_date).date()),
        "exit_date": str(pd.Timestamp(exit_date).date()),
        "entry_lead_used": int(entry_lead_used),
        "exit_offset_used": int(exit_offset_used),
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
    """Recompute gate membership from stored rows (pnl is k/lookback-independent;
    coverage snapping depends only on data coverage, never on k/lookback, so the
    invariant survives it). ``lookback`` values above ``_MAX_LOOKBACK`` (20) are
    effectively capped — stored ``prior_moves`` are trimmed to the last 20."""
    fired: list[float] = []
    for ev in events:
        window = ev["prior_moves"][-lookback:]
        expected = float(pd.Series(window).mean()) if window else float("nan")
        if _SIG.passes_filter(expected, ev["implied_move"], k):
            fired.append(float(ev["pnl"]))
    return fired


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/inf) with None so the results
    file is strict JSON (json.dumps emits bare NaN/Infinity tokens otherwise,
    which jq / JSON.parse reject)."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
