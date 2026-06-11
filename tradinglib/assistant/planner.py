# tradinglib/assistant/planner.py
"""Data plumbing for the options-planner chat tools.

``propose_levels`` and ``hypothesis_ticket`` gather bars/chain/earnings and
drive the strategist pipeline. They raise on unusable inputs; the tool
dispatchers in tools.py turn that into ``(message, is_error=True)`` so the
model can self-correct. Loaders use ``refresh=True``: on the deployed volume
the parquet caches are frozen at first fetch, and a chat ticket priced off
stale bars would be silently wrong (same gotcha as the nightly ledger).

``propose_levels`` is the guided step: it computes candidate level scenarios
(ATR- and structure-grounded), 20-day support/resistance context, upcoming
ticker events (earnings, ex-dividend) with warnings, and a sparkline the
console renders as a chart card — so the dialogue proposes levels instead of
asking the user for them.
"""

from __future__ import annotations

import pandas as pd

from tradinglib.features.technical import atr
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.dividends import get_next_ex_dividend
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.loaders.options.yf_chain import fetch_chain
from tradinglib.strategist import build_hypothesis_ticket
from tradinglib.tournament.levels import _ATR_MULT, _ATR_WINDOW, direction, two_r_target

_BARS_LOOKBACK_DAYS = 270  # ~9 calendar months of dailies for ATR(14) + realized vol
_EARNINGS_WARN_DAYS = 14  # mirrors the scanner config default; ex-div uses it too
_SPARK_SESSIONS = 60  # sparkline length on the levels card
_SR_WINDOW = 20  # support/resistance lookback (sessions)
_TIGHT_ATR_MULT = 1.5
_WIDE_BAND_ATR_MULT = 2.5
_SR_STOP_BUFFER_ATR = 0.25  # structure stop sits this far beyond the swing
_STRUCT_STOP_MIN_ATR = 0.75  # structure scenario gates: stop distance in ATRs…
_STRUCT_STOP_MAX_ATR = 3.5  # …and reward >= 1R (see _directional_scenarios)


def _directional_scenarios(
    spot: float, atr14: float, high_20d: float, low_20d: float, stance: str
) -> list[dict]:
    """Candidate entry/stop/target sets for a long/short hypothesis.

    ``balanced`` is the tournament convention and always recommended;
    ``structure`` anchors on the 20-day swing and is offered only when its
    geometry earns it (stop distance within sane ATR bounds, reward >= 1R).
    """
    d = direction(stance)

    def levels(stop: float, target: float) -> dict:
        return {
            "entry": round(spot, 2),
            "entry_type": "market",
            "stop": round(stop, 2),
            "target": round(target, 2),
        }

    stop_bal = spot - d * _ATR_MULT * atr14
    stop_tight = spot - d * _TIGHT_ATR_MULT * atr14
    scenarios = [
        {
            "key": "balanced",
            "levels": levels(stop_bal, two_r_target(spot, stop_bal)),
            "note": "2x ATR(14) stop, 2R target — the tournament default, room for normal noise",
            "recommended": True,
        },
        {
            "key": "tight",
            "levels": levels(stop_tight, two_r_target(spot, stop_tight)),
            "note": "1.5x ATR stop — less risk per share buys more contracts, but whipsaw-prone",
            "recommended": False,
        },
    ]
    swing = low_20d if d > 0 else high_20d
    stop_struct = swing - d * _SR_STOP_BUFFER_ATR * atr14
    target_struct = high_20d if d > 0 else low_20d
    stop_dist = d * (spot - stop_struct)
    reward = d * (target_struct - spot)
    if _STRUCT_STOP_MIN_ATR * atr14 <= stop_dist <= _STRUCT_STOP_MAX_ATR * atr14 and (
        reward >= stop_dist
    ):
        side = "low" if d > 0 else "high"
        scenarios.append(
            {
                "key": "structure",
                "levels": levels(stop_struct, target_struct),
                "note": (
                    f"stop beyond the 20-day swing {side} — price there invalidates "
                    f"the thesis, not the noise; target at the opposite 20-day extreme"
                ),
                "recommended": False,
            }
        )
    return scenarios


def _neutral_scenarios(spot: float, atr14: float, high_20d: float, low_20d: float) -> list[dict]:
    """Candidate lower/upper bands for a range-bound hypothesis. ``range`` (the
    observed 20-day extremes) is offered only when it brackets spot with at
    least 1 ATR of room each side."""

    def band(mult: float) -> dict:
        return {"lower": round(spot - mult * atr14, 2), "upper": round(spot + mult * atr14, 2)}

    scenarios = [
        {
            "key": "balanced",
            "levels": band(_ATR_MULT),
            "note": "spot -/+ 2x ATR(14) — the tournament band default",
            "recommended": True,
        },
        {
            "key": "wide",
            "levels": band(_WIDE_BAND_ATR_MULT),
            "note": "half an ATR more room each side — higher hold odds, less credit",
            "recommended": False,
        },
    ]
    if low_20d < spot < high_20d and min(spot - low_20d, high_20d - spot) >= atr14:
        scenarios.append(
            {
                "key": "range",
                "levels": {"lower": round(low_20d, 2), "upper": round(high_20d, 2)},
                "note": "the observed 20-day range",
                "recommended": False,
            }
        )
    return scenarios


def _next_earnings(ticker: str, now: pd.Timestamp) -> tuple[pd.Timestamp, str] | None:
    """Earliest scheduled earnings after ``now`` as (UTC-aware ts, session),
    or None. Raises on fetch errors — callers isolate."""
    earnings = get_earnings_dates([ticker], refresh=True)
    when = pd.DatetimeIndex(earnings["earnings_datetime"])
    if when.tz is None:
        when = when.tz_localize("UTC")
    future = earnings.loc[when > now].assign(_when=when[when > now]).sort_values("_when")
    if future.empty:
        return None
    row = future.iloc[0]
    return row["_when"], str(row["session"])


def _days_until(when_naive: pd.Timestamp, now: pd.Timestamp) -> int:
    """Calendar days from ``now`` (UTC-aware) to a naive date-resolution ts."""
    today = now.tz_convert("UTC").tz_localize(None).normalize()
    return int((when_naive.normalize() - today).days)


def _upcoming_events(ticker: str, now: pd.Timestamp) -> dict:
    """Next earnings + ex-dividend with <= 14d warnings. Either fetch failing
    appends a note and is otherwise non-fatal (mirrors the ticket path)."""
    events: dict = {"next_earnings": None, "ex_dividend": None, "warnings": [], "notes": []}
    try:
        found = _next_earnings(ticker, now)
        if found is not None:
            when, session = found
            when_naive = when.tz_convert("UTC").tz_localize(None)
            days = _days_until(when_naive, now)
            date = when_naive.strftime("%Y-%m-%d")
            events["next_earnings"] = {"date": date, "days_until": days, "session": session}
            if days <= _EARNINGS_WARN_DAYS:
                events["warnings"].append(
                    f"earnings {date} ({session}) in {days}d — gap risk, and IV is "
                    "typically inflated into the print"
                )
    except Exception:
        events["notes"].append("earnings lookup failed; earnings risk unchecked")
    try:
        ex_div = get_next_ex_dividend(ticker)
        if ex_div is not None:
            days = _days_until(ex_div, now)
            date = ex_div.strftime("%Y-%m-%d")
            events["ex_dividend"] = {"date": date, "days_until": days}
            if days <= _EARNINGS_WARN_DAYS:
                events["warnings"].append(
                    f"ex-dividend {date} in {days}d — short calls risk early assignment; "
                    "the underlying drops by the dividend"
                )
    except Exception:
        events["notes"].append("ex-dividend lookup failed")
    return events


def propose_levels(ticker: str, stance: str) -> dict:
    """Level scenarios + chart context + events for a hypothesis.

    ``levels``/``method`` describe the recommended scenario (the tournament
    conventions: directional entry = last close, stop = 2x ATR(14), target =
    2R; neutral band = spot -/+ 2x ATR(14)). ``scenarios`` carries the
    alternatives the user picks from, ``sparkline``/``context`` feed the
    console's chart card, and ``events.warnings`` flag earnings/ex-div inside
    14 days.
    """
    ticker = ticker.upper()
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=_BARS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    bars = load_daily(ticker, start=start, refresh=True)
    if len(bars) == 0:
        raise ValueError(f"no daily bars for {ticker!r} — is the ticker valid?")
    spot = float(bars["close"].iloc[-1])
    atr14 = float(atr(bars["high"], bars["low"], bars["close"], _ATR_WINDOW).iloc[-1])
    if not atr14 > 0.0:
        raise ValueError(f"degenerate ATR ({atr14}); cannot place levels")
    high_20d = float(bars["high"].iloc[-_SR_WINDOW:].max())
    low_20d = float(bars["low"].iloc[-_SR_WINDOW:].min())

    if stance == "neutral":
        scenarios = _neutral_scenarios(spot, atr14, high_20d, low_20d)
        method = "band = spot -/+ 2x ATR(14); structures sell premium outside the band"
    else:
        scenarios = _directional_scenarios(spot, atr14, high_20d, low_20d, stance)
        method = "entry = last close; stop = 2x ATR(14); target = 2R (entry-to-stop distance)"
    recommended = next(s for s in scenarios if s["recommended"])

    closes = bars["close"].iloc[-_SPARK_SESSIONS:]
    return {
        "ticker": ticker,
        "stance": stance,
        "levels": dict(recommended["levels"]),
        "spot": round(spot, 2),
        "atr14": round(atr14, 2),
        "asof": bars.index[-1].strftime("%Y-%m-%d"),
        "method": method,
        "scenarios": scenarios,
        "context": {"high_20d": round(high_20d, 2), "low_20d": round(low_20d, 2)},
        "events": _upcoming_events(ticker, now),
        "sparkline": {
            "start": closes.index[0].strftime("%Y-%m-%d"),
            "end": closes.index[-1].strftime("%Y-%m-%d"),
            "closes": [round(float(c), 2) for c in closes],
        },
    }


def hypothesis_ticket(
    *,
    ticker: str,
    stance: str,
    levels: dict,
    account_size: float,
    risk_per_trade_pct: float,
    preference: str = "auto",
    hypothesis: str = "",
    structure_key: str | None = None,
) -> dict:
    """Live-data ticket for a user hypothesis. Raises on bar/chain failures
    (the dispatcher relays them); an earnings-fetch failure only loses the
    earnings demotion, mirroring the scan pipeline's isolation. Pass
    ``structure_key`` to pin the recommendation and exit plan to one candidate."""
    ticker = ticker.upper()
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=_BARS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    bars = load_daily(ticker, start=start, refresh=True)
    if len(bars) == 0:
        raise ValueError(f"no daily bars for {ticker!r} — is the ticker valid?")
    chain = fetch_chain(ticker)

    next_earnings = None
    earnings_failed = False
    try:
        found = _next_earnings(ticker, now)
        if found is not None:
            next_earnings = found[0]
    except Exception:  # non-fatal: the ticket just loses the earnings demotion
        earnings_failed = True
    earnings_warning = bool(
        next_earnings is not None and next_earnings <= now + pd.Timedelta(days=_EARNINGS_WARN_DAYS)
    )

    full_levels = {
        **levels,
        "condition": hypothesis or f"user hypothesis: {stance} {ticker}",
    }
    ticket = build_hypothesis_ticket(
        ticker=ticker,
        stance=stance,
        levels=full_levels,
        bars=bars,
        chain=chain,
        preference=preference,
        next_earnings=next_earnings,
        earnings_warning=earnings_warning,
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
        structure_key=structure_key,
    )
    if earnings_failed:
        ticket["warnings"].append("earnings lookup failed; earnings risk unchecked")
    for s in ticket["structures"]:
        s["calculator_url"] = optionstrat_url(ticket["ticker"], s)
    return ticket


_OPTIONSTRAT_BASE = "https://optionstrat.com/build/custom"


def optionstrat_url(ticker: str, structure: dict) -> str | None:
    """Prefilled OptionStrat profit-calculator link for an option structure.

    One ``[-].<TICKER><YYMMDD><C|P><strike>x<qty>@<mid>`` token per leg
    (``-`` marks a short leg), comma-joined. Pure string assembly — no
    network. Stock plans have no legs and get no link.
    """
    legs = structure.get("legs") or []
    if not legs:
        return None
    qty = structure.get("quantity") or 1  # unsized still gets a 1-lot preview
    tokens = []
    for leg in legs:
        exp = pd.Timestamp(leg["expiration"]).strftime("%y%m%d")
        sign = "-" if leg["action"] == "sell" else ""
        right = "C" if leg["right"] == "call" else "P"
        strike = f"{leg['strike']:g}"  # 95.0 -> "95", 90.5 -> "90.5"
        tokens.append(f"{sign}.{ticker}{exp}{right}{strike}x{qty}@{leg['mid']:.2f}")
    return f"{_OPTIONSTRAT_BASE}/{ticker}/{','.join(tokens)}"
