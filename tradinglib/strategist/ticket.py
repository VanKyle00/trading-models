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
build_hypothesis_ticket is the chat path: same core, user-confirmed levels, no
tournament evidence, and options-only — the stock plan is never offered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.options.surface import realized_vol
from tradinglib.strategist.exit_plan import build_exit_plan
from tradinglib.strategist.quotes import atm_iv
from tradinglib.strategist.sizing import size_structure
from tradinglib.strategist.structures import (
    Band,
    Structure,
    build_chat_structures,
    build_neutral_structures,
    build_structures,
    structure_theta_week,
)
from tradinglib.tournament.levels import Levels
from tradinglib.tournament.strategies import STRATEGIES

IV_SELL_PREMIUM = 1.2
IV_BUY_PREMIUM = 0.9
INDICATIVE_WARNING = (
    "indicative quotes: last/close marks as of quotes_asof; re-check at the open, use limit orders"
)

_UNDEFINED_RISK = {"stock", "stock_short", "csp"}

_ORDER = {
    ("long", "directional"): ["stock", "long_call", "bull_put_spread", "csp"],
    ("long", "premium"): ["bull_put_spread", "csp", "stock", "long_call"],
    ("short", "directional"): ["long_put", "stock_short", "bear_call_spread"],
    ("short", "premium"): ["bear_call_spread", "long_put", "stock_short"],
}

# debit spreads sort (and recommend) as their long-option family: the spread is
# the family representative whenever its >35% cost-cut rule built it
_FAMILY = {"call_debit_spread": "long_call", "put_debit_spread": "long_put"}


def _preference(premium_first: bool, stance: str, iv_ratio: float | None) -> list[str]:
    if iv_ratio is not None:
        if iv_ratio > IV_SELL_PREMIUM:
            premium_first = True
        elif iv_ratio < IV_BUY_PREMIUM:
            premium_first = False
    return _ORDER[(stance, "premium" if premium_first else "directional")]


def _reason_directional(
    s: Structure,
    *,
    iv_ratio: float | None,
    use_iv_override: bool,
    premium_first: bool,
    fallthrough: bool = False,
) -> str:
    if fallthrough:
        base = "first candidate to size within the risk budget"
    else:
        base = ""
        if use_iv_override and iv_ratio is not None:
            if iv_ratio > IV_SELL_PREMIUM:
                base = f"IV/RV {iv_ratio:.2f} > {IV_SELL_PREMIUM} favors selling premium"
            elif iv_ratio < IV_BUY_PREMIUM:
                base = f"IV/RV {iv_ratio:.2f} < {IV_BUY_PREMIUM} favors buying premium"
        if not base:
            base = "premium-selling preference" if premium_first else "directional preference"
    if s.kind in _FAMILY:
        return base + "; the short leg at the target cuts the long option's cost > 35%"
    if s.key.endswith("_d65"):
        return base + "; ~0.65 delta balances cost vs probability"
    return base


_NEUTRAL_REASONS = {
    "condor": "shorts just beyond your band — best credit-for-room balance",
    "condor_wide": "shorts one strike further out — more room, less credit",
    "butterfly": "most credit, narrowest profit zone",
}


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    # unlike quotes._naive this converts to UTC first: an aware earnings timestamp
    # must compare against expiration DATES as the UTC instant's calendar date
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


def _assemble(
    *,
    stance: str,
    levels: Levels,
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    premium_first: bool,
    use_iv_override: bool,
    include_stock: bool = True,
    expand: bool = False,
    next_earnings: pd.Timestamp | None,
    earnings_warning: bool,
    account_size: float,
    risk_per_trade_pct: float,
) -> tuple[list[Structure], list[str], float | None, pd.Timestamp, bool, float]:
    """Shared core: build -> order -> demote -> size -> recommend.

    Returns ``(structures, warnings, iv_ratio, asof, has_chain, spot)``.
    ``iv_ratio`` is always computed for framing; ``use_iv_override=False`` keeps
    it out of the ordering decision (a pinned chat preference). When
    ``expand=True``, the chat candidate ladder (build_chat_structures) replaces
    the single long_option builder so all delta rungs are offered.
    """
    has_chain = len(chain) > 0
    spot = float(chain["spot"].iloc[0]) if has_chain else float(bars["close"].iloc[-1])
    asof = (
        pd.Timestamp(chain["date"].iloc[0]) if has_chain else _naive(pd.Timestamp(bars.index[-1]))
    )

    if expand:
        structures, drop_notes = build_chat_structures(chain, levels, stance, spot=spot, asof=asof)
    else:
        structures = build_structures(
            chain, levels, stance, spot=spot, asof=asof, include_stock=include_stock
        )
        drop_notes = []
    warnings = [INDICATIVE_WARNING, *drop_notes]
    if not any(s.unit == "contract" for s in structures):
        # every option structure failed the gate (or no chain)
        if not include_stock:  # chat path: nothing to fall back on — surface it
            raise ValueError("no option structure passed the liquidity gate")
        warnings.append("options_illiquid: no option structure passed the liquidity gate")

    rv = float(realized_vol(bars["close"]).iloc[-1])
    chain_iv = atm_iv(chain, spot=spot, asof=asof) if has_chain else None
    iv_ratio = float(chain_iv / rv) if chain_iv is not None and np.isfinite(rv) and rv > 0 else None

    order = _preference(premium_first, stance, iv_ratio if use_iv_override else None)
    structures.sort(key=lambda s: order.index(_FAMILY.get(s.kind, s.kind)))
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
    for s in structures:
        s.theta_week = structure_theta_week(s, spot=spot)
    recommended.reason = _reason_directional(
        recommended,
        iv_ratio=iv_ratio,
        use_iv_override=use_iv_override,
        premium_first=premium_first,
        fallthrough=recommended is not structures[0],
    )
    return structures, warnings, iv_ratio, asof, has_chain, spot


def _assemble_neutral(
    *,
    band: Band,
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    next_earnings: pd.Timestamp | None,
    earnings_warning: bool,
    account_size: float,
    risk_per_trade_pct: float,
) -> tuple[list[Structure], list[str], float | None, pd.Timestamp, bool, float]:
    """Neutral core: condor/butterfly only. Both are defined-risk so the
    earnings demotion doesn't apply — an expiry spanning earnings gets a
    structure warning instead (a gap can invalidate the range thesis).

    Returns ``(structures, warnings, iv_ratio, asof, has_chain, spot)``.
    """
    has_chain = len(chain) > 0
    spot = float(chain["spot"].iloc[0]) if has_chain else float(bars["close"].iloc[-1])
    asof = (
        pd.Timestamp(chain["date"].iloc[0]) if has_chain else _naive(pd.Timestamp(bars.index[-1]))
    )

    structures = build_neutral_structures(chain, band, spot=spot, asof=asof)
    if not structures:
        raise ValueError("no option structure passed the liquidity gate")
    warnings = [INDICATIVE_WARNING]

    rv = float(realized_vol(bars["close"]).iloc[-1])
    chain_iv = atm_iv(chain, spot=spot, asof=asof) if has_chain else None
    iv_ratio = float(chain_iv / rv) if chain_iv is not None and np.isfinite(rv) and rv > 0 else None

    if earnings_warning:
        ne = _naive(next_earnings) if next_earnings is not None else None
        for s in structures:
            if ne is None or any(pd.Timestamp(leg["expiration"]) >= ne for leg in s.legs):
                s.warnings.append("short-vol structure spans earnings; a gap can break the range")
        when = f" ({ne:%Y-%m-%d})" if ne is not None else ""
        warnings.append(f"earnings inside the warn window{when}")

    for s in structures:
        size_structure(s, account_size=account_size, risk_per_trade_pct=risk_per_trade_pct)
    recommended = next((s for s in structures if s.quantity), structures[0])
    recommended.recommended = True
    for s in structures:
        s.theta_week = structure_theta_week(s, spot=spot)
    recommended.reason = _NEUTRAL_REASONS.get(recommended.key, "first range structure to size")
    return structures, warnings, iv_ratio, asof, has_chain, spot


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
    structures, warnings, iv_ratio, asof, has_chain, _spot = _assemble(
        stance=stance,
        levels=levels,
        bars=bars,
        chain=chain,
        premium_first=sdef.style == "mean_reversion",
        use_iv_override=True,
        next_earnings=next_earnings,
        earnings_warning=earnings_warning,
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
    )
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


def build_hypothesis_ticket(
    *,
    ticker: str,
    stance: str,
    levels: dict,
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    preference: str = "auto",
    structure_key: str | None = None,
    next_earnings: pd.Timestamp | None = None,
    earnings_warning: bool = False,
    account_size: float = 100_000.0,
    risk_per_trade_pct: float = 0.01,
) -> dict:
    """A chat-built ticket: user hypothesis + confirmed levels, no tournament
    evidence, options-only. ``preference`` pins the ordering ("directional" |
    "premium") or lets the IV/RV tilt decide ("auto") — ignored for the
    "neutral" stance, whose levels are a {lower, upper, condition} band and
    whose structures (iron condor, iron butterfly) all sell premium.
    ``structure_key`` pins the recommendation (and the exit plan) to one
    candidate from a prior build; an unknown key raises with the valid keys
    listed. Pure; no I/O."""
    if preference not in ("auto", "directional", "premium"):
        raise ValueError(f"preference must be auto|directional|premium, got {preference!r}")
    levels_obj: Levels | Band
    if stance == "neutral":
        band = Band(**levels)
        structures, warnings, iv_ratio, asof, has_chain, _spot = _assemble_neutral(
            band=band,
            bars=bars,
            chain=chain,
            next_earnings=next_earnings,
            earnings_warning=earnings_warning,
            account_size=account_size,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        levels_obj = band
    else:
        lvls = Levels(**levels)
        structures, warnings, iv_ratio, asof, has_chain, _spot = _assemble(
            stance=stance,
            levels=lvls,
            bars=bars,
            chain=chain,
            premium_first=preference == "premium",
            use_iv_override=preference == "auto",
            include_stock=False,
            expand=True,
            next_earnings=next_earnings,
            earnings_warning=earnings_warning,
            account_size=account_size,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        levels_obj = lvls
    if structure_key is not None:
        chosen = next((s for s in structures if s.key == structure_key), None)
        if chosen is None:
            valid = ", ".join(s.key for s in structures)
            raise ValueError(f"unknown structure_key {structure_key!r}; valid keys: {valid}")
        for s in structures:
            s.recommended = False
            s.reason = None
        chosen.recommended = True
        chosen.reason = "pinned by user request"
    recommended = next(s for s in structures if s.recommended)
    plan = build_exit_plan(recommended, levels_obj, next_earnings=next_earnings)
    return {
        "source": "chat",
        "ticker": ticker,
        "stance": stance,
        "levels": dict(levels),
        "quotes_asof": asof.strftime("%Y-%m-%d") if has_chain else None,
        "iv_ratio": iv_ratio,
        "next_earnings": (
            _naive(next_earnings).strftime("%Y-%m-%d") if next_earnings is not None else None
        ),
        "account_size": account_size,
        "risk_per_trade_pct": risk_per_trade_pct,
        "structures": [s.as_dict() for s in structures],
        "plan": plan.as_dict() if plan is not None else None,
        "warnings": warnings,
    }
