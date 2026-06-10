"""Ticket assembly: structures -> deterministic selection -> sizing -> framing.

Selection (spec C3): the winner's style sets the base preference —
trend/breakout favors directional convexity, mean reversion favors premium
selling — then the IV/realized-vol ratio overrides it (> 1.2 sell premium,
< 0.9 buy premium), and an earnings date inside a structure's tenor demotes
undefined-risk structures (CSP, stock) below defined-risk spreads. Exactly one
built structure is ``recommended``: the first in the final order that sized to
at least one unit, else the first overall (carrying its ``unsized`` warning).
The ticket never asserts expected profitability (C5): quotes are indicative
marks, PoP is the market's own number, and the OOS evidence rides along.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.options.surface import realized_vol
from tradinglib.strategist.quotes import atm_iv
from tradinglib.strategist.sizing import size_structure
from tradinglib.strategist.structures import Structure, build_structures
from tradinglib.tournament.levels import Levels
from tradinglib.tournament.strategies import STRATEGIES

IV_SELL_PREMIUM = 1.2
IV_BUY_PREMIUM = 0.9
INDICATIVE_WARNING = (
    "indicative quotes: last/close marks as of quotes_asof; re-check at the open, use limit orders"
)

_UNDEFINED_RISK = {"stock", "stock_short", "csp"}

_ORDER = {
    ("long", "directional"): ["stock", "long_call", "call_debit_spread", "bull_put_spread", "csp"],
    ("long", "premium"): ["bull_put_spread", "csp", "stock", "long_call", "call_debit_spread"],
    ("short", "directional"): ["long_put", "put_debit_spread", "stock_short", "bear_call_spread"],
    ("short", "premium"): ["bear_call_spread", "long_put", "put_debit_spread", "stock_short"],
}


def _preference(style: str, stance: str, iv_ratio: float | None) -> list[str]:
    premium_first = style == "mean_reversion"
    if iv_ratio is not None:
        if iv_ratio > IV_SELL_PREMIUM:
            premium_first = True
        elif iv_ratio < IV_BUY_PREMIUM:
            premium_first = False
    return _ORDER[(stance, "premium" if premium_first else "directional")]


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def _spans_earnings(s: Structure, next_earnings: pd.Timestamp | None) -> bool:
    """Undefined-risk structures held across (or with no expiry to cap) earnings."""
    if s.kind not in _UNDEFINED_RISK:
        return False
    if not s.legs:  # stock plans have no expiry bounding the hold
        return True
    if next_earnings is None:
        return True
    ne = _naive(next_earnings)
    return any(pd.Timestamp(leg["expiration"]) >= ne for leg in s.legs)


def build_ticket(
    *,
    ticker: str,
    stance: str,
    winner: dict,
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    fa: dict | None = None,
    winner_changed: bool | None = None,
    next_earnings: pd.Timestamp | None = None,
    earnings_warning: bool = False,
    account_size: float = 100_000.0,
    risk_per_trade_pct: float = 0.01,
) -> dict:
    """One report-ready trade ticket for a tournament winner. Pure; no I/O."""
    sdef = STRATEGIES[winner["strategy"]]
    levels = Levels(**winner["levels"])
    has_chain = len(chain) > 0
    spot = float(chain["spot"].iloc[0]) if has_chain else float(bars["close"].iloc[-1])
    asof = (
        pd.Timestamp(chain["date"].iloc[0]) if has_chain else _naive(pd.Timestamp(bars.index[-1]))
    )

    structures = build_structures(chain, levels, stance, spot=spot, asof=asof)
    warnings = [INDICATIVE_WARNING]
    if len(structures) == 1:  # every option structure failed the gate (or no chain)
        warnings.append("options_illiquid: no option structure passed the liquidity gate")

    rv = float(realized_vol(bars["close"]).iloc[-1])
    chain_iv = atm_iv(chain, spot=spot, asof=asof) if has_chain else None
    iv_ratio = float(chain_iv / rv) if chain_iv is not None and np.isfinite(rv) and rv > 0 else None

    order = _preference(sdef.style, stance, iv_ratio)
    structures.sort(key=lambda s: order.index(s.kind))
    if earnings_warning:
        kept = [s for s in structures if not _spans_earnings(s, next_earnings)]
        demoted = [s for s in structures if _spans_earnings(s, next_earnings)]
        for s in demoted:
            s.warnings.append("undefined risk across earnings; demoted below defined-risk spreads")
        structures = kept + demoted
        when = f" ({_naive(next_earnings):%Y-%m-%d})" if next_earnings is not None else ""
        warnings.append(f"earnings inside the warn window{when}")

    for s in structures:
        size_structure(s, account_size=account_size, risk_per_trade_pct=risk_per_trade_pct)
    recommended = next((s for s in structures if s.quantity), structures[0])
    recommended.recommended = True

    return {
        "ticker": ticker,
        "stance": stance,
        "strategy": winner["strategy"],
        "style": sdef.style,
        "params": winner["params"],
        "levels": dict(winner["levels"]),
        "evidence": {
            "deflated_sharpe": winner["deflated_sharpe"],
            "sharpe": winner["sharpe"],
            "n_trades": winner["n_trades"],
            "n_windows": winner["n_windows"],
            "winner_changed": winner_changed,
            "fa_rank": fa["rank"] if fa else None,
            "fa_score": fa["fa_score"] if fa else None,
        },
        "quotes_asof": asof.strftime("%Y-%m-%d") if has_chain else None,
        "iv_ratio": iv_ratio,
        "next_earnings": (
            _naive(next_earnings).strftime("%Y-%m-%d") if next_earnings is not None else None
        ),
        "account_size": account_size,
        "risk_per_trade_pct": risk_per_trade_pct,
        "structures": [s.as_dict() for s in structures],
        "warnings": warnings,
    }
