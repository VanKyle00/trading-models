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
from tradinglib.tournament.levels import protective_stop, two_r_target

_BARS_LOOKBACK_DAYS = 270  # ~9 calendar months of dailies for ATR(14) + realized vol
_ATR_WINDOW = 14


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
