"""Option-leg quoting: chain slicing, the liquidity gate, strike/expiry pickers.

The playbook never builds a leg that fails the liquidity gate: ``bid > 0``,
mid-spread <= 10% of mid, open interest >= 100 — plus a finite IV > 0, since
delta-based strike selection and the market-implied PoP are meaningless
without a usable vol. Quotes are indicative last/close marks (framing, not
edge); deltas are Black-Scholes from the chain's own IV at zero rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradinglib.options.pricing import bs_greeks

MAX_SPREAD_FRAC = 0.10
MIN_OPEN_INTEREST = 100.0


@dataclass(frozen=True)
class LegQuote:
    """One gate-passing option quote, ready to become a structure leg."""

    expiration: pd.Timestamp
    strike: float
    right: str  # "call" | "put"
    bid: float
    ask: float
    mid: float
    iv: float
    delta: float
    dte: int


def years(dte: int) -> float:
    """Calendar DTE -> years, floored at one day so t > 0 everywhere."""
    return max(dte, 1) / 365.0


def liquid_quotes(
    chain: pd.DataFrame, *, right: str, expiration: pd.Timestamp, spot: float, asof: pd.Timestamp
) -> list[LegQuote]:
    """Gate-passing quotes for one (right, expiration), sorted by strike."""
    sub = chain[(chain["right"] == right) & (chain["expiration"] == expiration)]
    dte = int((pd.Timestamp(expiration) - pd.Timestamp(asof)).days)
    quotes: list[LegQuote] = []
    for row in sub.itertuples():
        bid, ask = float(row.bid), float(row.ask)
        if not (bid > 0.0 and ask >= bid):
            continue
        mid = (bid + ask) / 2.0
        if (ask - bid) / mid > MAX_SPREAD_FRAC:
            continue
        if not (pd.notna(row.open_interest) and float(row.open_interest) >= MIN_OPEN_INTEREST):
            continue
        if not (pd.notna(row.iv) and float(row.iv) > 0.0):
            continue
        strike, iv = float(row.strike), float(row.iv)
        delta = bs_greeks(right, spot, strike, years(dte), iv, 0.0).delta  # type: ignore[arg-type]
        quotes.append(
            LegQuote(
                expiration=pd.Timestamp(expiration),
                strike=strike,
                right=right,
                bid=bid,
                ask=ask,
                mid=mid,
                iv=iv,
                delta=float(delta),
                dte=dte,
            )
        )
    return sorted(quotes, key=lambda q: q.strike)


def pick_expiration(
    chain: pd.DataFrame, *, dte_lo: int, dte_hi: int, asof: pd.Timestamp
) -> pd.Timestamp | None:
    """The listed expiration nearest the DTE-window midpoint; None if none listed."""
    if chain.empty:
        return None
    midpoint = (dte_lo + dte_hi) / 2.0
    best: pd.Timestamp | None = None
    best_gap = float("inf")
    for exp in pd.DatetimeIndex(chain["expiration"].unique()):
        dte = (exp - pd.Timestamp(asof)).days
        if not dte_lo <= dte <= dte_hi:
            continue
        gap = abs(dte - midpoint)
        if gap < best_gap:
            best, best_gap = exp, gap
    return best


def by_delta(quotes: list[LegQuote], target: float) -> LegQuote | None:
    return min(quotes, key=lambda q: abs(q.delta - target)) if quotes else None


def at_or_below(quotes: list[LegQuote], level: float) -> LegQuote | None:
    below = [q for q in quotes if q.strike <= level]
    return max(below, key=lambda q: q.strike) if below else None


def strictly_below(quotes: list[LegQuote], level: float) -> LegQuote | None:
    below = [q for q in quotes if q.strike < level]
    return max(below, key=lambda q: q.strike) if below else None


def strictly_above(quotes: list[LegQuote], level: float) -> LegQuote | None:
    above = [q for q in quotes if q.strike > level]
    return min(above, key=lambda q: q.strike) if above else None


def nearest_strike(quotes: list[LegQuote], level: float) -> LegQuote | None:
    return min(quotes, key=lambda q: abs(q.strike - level)) if quotes else None


def atm_iv(
    chain: pd.DataFrame, *, spot: float, asof: pd.Timestamp, dte_lo: int = 30, dte_hi: int = 45
) -> float | None:
    """IV of the gate-passing quote nearest the spot in the income window (IV-tilt input)."""
    exp = pick_expiration(chain, dte_lo=dte_lo, dte_hi=dte_hi, asof=asof)
    if exp is None:
        return None
    quotes = liquid_quotes(chain, right="call", expiration=exp, spot=spot, asof=asof)
    quotes += liquid_quotes(chain, right="put", expiration=exp, spot=spot, asof=asof)
    q = nearest_strike(quotes, spot)
    return q.iv if q else None
