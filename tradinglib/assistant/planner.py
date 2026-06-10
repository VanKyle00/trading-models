# tradinglib/assistant/planner.py
"""Data plumbing for the options-planner chat tools.

``propose_levels`` and ``hypothesis_ticket`` gather bars/chain/earnings and
drive the strategist pipeline. They raise on unusable inputs; the tool
dispatchers in tools.py turn that into ``(message, is_error=True)`` so the
model can self-correct. Loaders use ``refresh=True``: on the deployed volume
the parquet caches are frozen at first fetch, and a chat ticket priced off
stale bars would be silently wrong (same gotcha as the nightly ledger).
"""

from __future__ import annotations

import pandas as pd

from tradinglib.features.technical import atr
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.loaders.options.yf_chain import fetch_chain
from tradinglib.strategist import build_hypothesis_ticket
from tradinglib.tournament.levels import _ATR_WINDOW, protective_stop, two_r_target

_BARS_LOOKBACK_DAYS = 270  # ~9 calendar months of dailies for ATR(14) + realized vol


def propose_levels(ticker: str, stance: str) -> dict:
    """Default levels for a hypothesis: entry = last close, stop = 2x ATR(14),
    target = 2R — the same conventions tournament rules without native exits use."""
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_BARS_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    bars = load_daily(ticker, start=start, refresh=True)
    if len(bars) == 0:
        raise ValueError(f"no daily bars for {ticker!r} — is the ticker valid?")
    entry = float(bars["close"].iloc[-1])
    stop = float(protective_stop(bars, entry, stance))
    target = float(two_r_target(entry, stop))
    atr14 = float(atr(bars["high"], bars["low"], bars["close"], _ATR_WINDOW).iloc[-1])
    return {
        "ticker": ticker.upper(),
        "stance": stance,
        "levels": {
            "entry": round(entry, 2),
            "entry_type": "market",
            "stop": round(stop, 2),
            "target": round(target, 2),
        },
        "spot": round(entry, 2),
        "atr14": round(atr14, 2),
        "asof": bars.index[-1].strftime("%Y-%m-%d"),
        "method": "entry = last close; stop = 2x ATR(14); target = 2R (entry-to-stop distance)",
    }


_EARNINGS_WARN_DAYS = 14  # mirrors the scanner config default


def hypothesis_ticket(
    *,
    ticker: str,
    stance: str,
    levels: dict,
    account_size: float,
    risk_per_trade_pct: float,
    preference: str = "auto",
    hypothesis: str = "",
) -> dict:
    """Live-data ticket for a user hypothesis. Raises on bar/chain failures
    (the dispatcher relays them); an earnings-fetch failure only loses the
    earnings demotion, mirroring the scan pipeline's isolation."""
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=_BARS_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    bars = load_daily(ticker, start=start, refresh=True)
    if len(bars) == 0:
        raise ValueError(f"no daily bars for {ticker!r} — is the ticker valid?")
    chain = fetch_chain(ticker)

    last_bar = pd.Timestamp(bars.index[-1])
    asof = last_bar.tz_localize("UTC") if last_bar.tzinfo is None else last_bar.tz_convert("UTC")
    next_earnings = None
    try:
        earnings = get_earnings_dates([ticker], refresh=True)
        dts = pd.DatetimeIndex(earnings["earnings_datetime"])
        if dts.tz is None:
            dts = dts.tz_localize("UTC")
        future = dts[dts > asof]
        if len(future):
            next_earnings = future.min()
    except Exception:  # non-fatal: the ticket just loses the earnings demotion
        next_earnings = None
    earnings_warning = bool(
        next_earnings is not None and next_earnings <= asof + pd.Timedelta(days=_EARNINGS_WARN_DAYS)
    )

    full_levels = {
        **levels,
        "condition": hypothesis or f"user hypothesis: {stance} {ticker.upper()}",
    }
    ticket = build_hypothesis_ticket(
        ticker=ticker.upper(),
        stance=stance,
        levels=full_levels,
        bars=bars,
        chain=chain,
        preference=preference,
        next_earnings=next_earnings,
        earnings_warning=earnings_warning,
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
    )
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
