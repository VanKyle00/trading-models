"""After-hours option-chain fallback (CBOE delayed-quotes CDN).

yfinance zeroes option bid/ask AND open interest outside regular trading
hours, so the planner's liquidity gate rejects everything overnight. CBOE's
public delayed-quotes JSON keeps the last session's closing quotes available
around the clock — good enough to PLAN against; fills must be re-verified at
the open. Same canonical ``CHAIN_COLUMNS`` schema as ``yf_chain.fetch_chain``;
in-memory only, never persisted.

Symbology: the request URL wants dots for share classes (``BRK.B`` — the
Yahoo-style ``BRK-B`` is translated; dashed forms return HTTP 403), while each
record's OCC-style ``option`` symbol strips the dot (``BRKB260612C00270000``).
Expiration/right/strike are parsed from the symbol's fixed 15-char tail, so
the variable-length root never needs decoding. CBOE is mocked in tests and
never called live (repo convention).
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

import pandas as pd

from tradinglib.loaders.options.yf_chain import CHAIN_COLUMNS, FETCH_MAX_DAYS

_CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
_OCC_TAIL = re.compile(r"(\d{6})([CP])(\d{8})$")  # yymmdd, right, strike*1000


def _get_json(url: str) -> dict[str, Any]:
    """Fetch and decode one CDN payload. Raises on HTTP/network failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_cboe_chain(ticker: str, *, max_days: int = FETCH_MAX_DAYS) -> pd.DataFrame:
    """In-memory near-term chain from CBOE delayed quotes (never persisted).

    Canonical snapshot schema plus ``open_interest``/``volume``, matching
    ``yf_chain.fetch_chain``. ``date`` is today's ET date so DTE math stays
    correct overnight; the quotes themselves are the last session's closing
    marks. Raises on fetch failure; malformed option symbols are skipped.
    CBOE lists the full board (LEAPS, just-expired series) — only expirations
    in ``[today, today + max_days]`` are kept.
    """
    ticker = ticker.upper()
    payload = _get_json(_CBOE_URL.format(symbol=ticker.replace("-", ".")))
    data = payload["data"]
    spot = float(data["current_price"])
    today = pd.Timestamp(pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))
    cutoff = today + pd.Timedelta(days=max_days)

    rows = []
    for opt in data["options"]:
        m = _OCC_TAIL.search(str(opt["option"]))
        if m is None:
            continue
        expiration = pd.Timestamp(f"20{m[1][:2]}-{m[1][2:4]}-{m[1][4:6]}")
        if not today <= expiration <= cutoff:
            continue
        rows.append(
            {
                "date": today,
                "ticker": ticker,
                "expiration": expiration,
                "strike": int(m[3]) / 1000.0,
                "right": "call" if m[2] == "C" else "put",
                "bid": pd.to_numeric(opt.get("bid"), errors="coerce"),
                "ask": pd.to_numeric(opt.get("ask"), errors="coerce"),
                "iv": pd.to_numeric(opt.get("iv"), errors="coerce"),
                "spot": spot,
                "open_interest": pd.to_numeric(opt.get("open_interest"), errors="coerce"),
                "volume": pd.to_numeric(opt.get("volume"), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=CHAIN_COLUMNS)
    out = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    return out.sort_values(["expiration", "strike", "right"]).reset_index(drop=True)
