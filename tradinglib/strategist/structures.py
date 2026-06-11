"""Stock/option structure builders and payoff math for one ticket (spec C3).

Builders are pure: (chain, Levels, spot, asof) -> ``Structure | None``; None
means "not offered" — no expiration in the DTE window, a leg failed the
liquidity gate, or a credit rule failed. All price-space fields are PER SHARE;
``unit`` ("share" | "contract") tells sizing the x100 multiplier. The stock
plan's ``max_loss`` is the plan's risk to the stop — a gap through the stop
exceeds it. ``pop_market_implied`` is the market's zero-rate lognormal number
from the chain IV of the leg nearest the breakeven — the market's probability,
never an edge claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import pandas as pd
from scipy.stats import norm

from tradinglib.strategist.quotes import (
    LegQuote,
    at_or_below,
    by_delta,
    liquid_quotes,
    nearest_strike,
    pick_expiration,
    strictly_above,
    strictly_below,
    years,
)
from tradinglib.tournament.levels import Levels, direction

DTE_DIRECTIONAL = (60, 90)
DTE_INCOME = (30, 45)
TARGET_DELTA = 0.65
SPREAD_SAVES_FRAC = 0.35  # debit spread offered only when it cuts cost by more than this
SPREAD_WIDTH = 5.0
MIN_CREDIT_FRAC = 1.0 / 3.0


@dataclass
class Structure:
    kind: str
    label: str
    unit: str  # "share" | "contract"
    legs: list[dict]
    premium: float | None  # per share; >0 = debit paid, <0 = credit received
    max_loss: float | None  # per share; stock plans: risk to the stop
    max_gain: float | None  # per share; None = uncapped
    breakeven: float | None
    pop_market_implied: float | None
    rr: float | None
    premium_yield: float | None
    warnings: list[str] = field(default_factory=list)
    recommended: bool = False
    quantity: int | None = None
    loss_at_stop: float | None = None  # CSP scenario P/L per share at the ticket stop (+ = loss)
    breakevens: list[float] | None = None  # two-sided structures: [lower, upper]; breakeven stays None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Band:
    """Range-hypothesis levels for a neutral stance (chat path only)."""

    lower: float
    upper: float
    condition: str


def pop_market_implied(
    *, spot: float, level: float, vol: float, t_years: float, profit_above: bool
) -> float:
    """P(S_T ends on the profitable side of ``level``), zero-rate lognormal."""
    if level <= 0.0:  # lognormal support is (0, inf): junk quotes can push a breakeven <= 0
        return 1.0 if profit_above else 0.0
    d = (math.log(level / spot) + 0.5 * vol * vol * t_years) / (vol * math.sqrt(t_years))
    p_below = float(norm.cdf(d))
    return 1.0 - p_below if profit_above else p_below


def pop_range_market_implied(
    *, spot: float, lower: float, upper: float, vol_lower: float, vol_upper: float, t_years: float
) -> float:
    """P(lower < S_T < upper), zero-rate lognormal, one tail per band edge.

    Each edge uses its nearest leg's vol so the number stays consistent with
    the single-level PoPs; clamped at 0 because two different vols can cross.
    """
    p_below_upper = pop_market_implied(
        spot=spot, level=upper, vol=vol_upper, t_years=t_years, profit_above=False
    )
    p_below_lower = pop_market_implied(
        spot=spot, level=lower, vol=vol_lower, t_years=t_years, profit_above=False
    )
    return max(0.0, p_below_upper - p_below_lower)


def _pop_vol(legs: list[LegQuote], breakeven: float) -> float:
    """The chain vol nearest the level that matters."""
    return min(legs, key=lambda q: abs(q.strike - breakeven)).iv


def _leg(action: str, q: LegQuote) -> dict:
    return {
        "action": action,
        "right": q.right,
        "strike": q.strike,
        "expiration": q.expiration.strftime("%Y-%m-%d"),
        "dte": q.dte,
        "mid": q.mid,
        "iv": q.iv,
        "delta": q.delta,
    }


def stock_plan(levels: Levels, stance: str) -> Structure:
    """The stock leg of the ticket: always offered, sized by stop distance."""
    short = direction(stance) < 0
    risk = abs(levels.entry - levels.stop)
    gain = abs(levels.target - levels.entry)
    return Structure(
        kind="stock_short" if short else "stock",
        label="short stock plan" if short else "stock plan",
        unit="share",
        legs=[],
        premium=None,
        max_loss=risk,
        max_gain=gain,
        breakeven=levels.entry,
        pop_market_implied=None,
        rr=gain / risk if risk > 0 else None,
        premium_yield=None,
        warnings=(
            [
                "borrow/squeeze risk; borrow cost not modeled; "
                "a gap through the stop exceeds max_loss"
            ]
            if short
            else []
        ),
    )


def long_option(
    chain: pd.DataFrame, levels: Levels, *, spot: float, asof: pd.Timestamp, stance: str
) -> Structure | None:
    """Long call (long stance) / long put (short stance), 60-90 DTE, ~0.65 delta —
    or the debit spread with the short leg at the target when it cuts cost > 35%."""
    d = direction(stance)
    right = "call" if d > 0 else "put"
    exp = pick_expiration(chain, dte_lo=DTE_DIRECTIONAL[0], dte_hi=DTE_DIRECTIONAL[1], asof=asof)
    if exp is None:
        return None
    quotes = liquid_quotes(chain, right=right, expiration=exp, spot=spot, asof=asof)
    leg = by_delta(quotes, d * TARGET_DELTA)
    if leg is None or leg.mid <= 0:
        return None
    t = years(leg.dte)
    toward_target = [q for q in quotes if d * (q.strike - leg.strike) > 0]
    short_leg = nearest_strike(toward_target, levels.target)
    if short_leg is not None:
        net = leg.mid - short_leg.mid
        # net <= 0 (inverted junk quotes) falls through to the plain long option
        if net > 0 and net <= (1.0 - SPREAD_SAVES_FRAC) * leg.mid:
            width = abs(short_leg.strike - leg.strike)
            breakeven = leg.strike + d * net
            return Structure(
                kind=f"{right}_debit_spread",
                label=f"{leg.strike:g}/{short_leg.strike:g} {right} debit spread",
                unit="contract",
                legs=[_leg("buy", leg), _leg("sell", short_leg)],
                premium=net,
                max_loss=net,
                max_gain=width - net,
                breakeven=breakeven,
                pop_market_implied=pop_market_implied(
                    spot=spot,
                    level=breakeven,
                    vol=_pop_vol([leg, short_leg], breakeven),
                    t_years=t,
                    profit_above=d > 0,
                ),
                rr=(width - net) / net,
                premium_yield=None,
            )
    breakeven = leg.strike + d * leg.mid
    return Structure(
        kind=f"long_{right}",
        label=f"{leg.strike:g} {right} (~{leg.delta:+.2f} delta)",
        unit="contract",
        legs=[_leg("buy", leg)],
        premium=leg.mid,
        max_loss=leg.mid,
        max_gain=None,
        breakeven=breakeven,
        pop_market_implied=pop_market_implied(
            spot=spot, level=breakeven, vol=leg.iv, t_years=t, profit_above=d > 0
        ),
        rr=None,
        premium_yield=None,
    )


def cash_secured_put(
    chain: pd.DataFrame, levels: Levels, *, spot: float, asof: pd.Timestamp
) -> Structure | None:
    """30-45 DTE short put, strike at or below the stop. Long stance only; capital-sized."""
    exp = pick_expiration(chain, dte_lo=DTE_INCOME[0], dte_hi=DTE_INCOME[1], asof=asof)
    if exp is None:
        return None
    puts = liquid_quotes(chain, right="put", expiration=exp, spot=spot, asof=asof)
    leg = at_or_below(puts, levels.stop)
    if leg is None or leg.mid <= 0:
        return None
    credit = leg.mid
    breakeven = leg.strike - credit
    return Structure(
        kind="csp",
        label=f"{leg.strike:g} cash-secured put",
        unit="contract",
        legs=[_leg("sell", leg)],
        premium=-credit,
        max_loss=breakeven,
        max_gain=credit,
        breakeven=breakeven,
        pop_market_implied=pop_market_implied(
            spot=spot, level=breakeven, vol=leg.iv, t_years=years(leg.dte), profit_above=True
        ),
        rr=credit / breakeven if breakeven > 0 else None,
        premium_yield=credit / leg.strike,
        warnings=["assignment risk: max_loss assumes the stock at zero"],
        loss_at_stop=max(leg.strike - levels.stop, 0.0) - credit,
    )


def credit_spread(
    chain: pd.DataFrame, levels: Levels, *, spot: float, asof: pd.Timestamp, stance: str
) -> Structure | None:
    """Bull put (long) / bear call (short): 30-45 DTE, ~$5 wide, short strike beyond
    the invalidation level (the stop), credit >= 1/3 of the width."""
    d = direction(stance)
    right = "put" if d > 0 else "call"
    exp = pick_expiration(chain, dte_lo=DTE_INCOME[0], dte_hi=DTE_INCOME[1], asof=asof)
    if exp is None:
        return None
    quotes = liquid_quotes(chain, right=right, expiration=exp, spot=spot, asof=asof)
    short_leg = (
        strictly_below(quotes, levels.stop) if d > 0 else strictly_above(quotes, levels.stop)
    )
    if short_leg is None:
        return None
    wings = [q for q in quotes if d * (short_leg.strike - q.strike) > 0]
    long_leg = nearest_strike(wings, short_leg.strike - d * SPREAD_WIDTH)
    if long_leg is None:
        return None
    width = abs(short_leg.strike - long_leg.strike)
    credit = short_leg.mid - long_leg.mid
    if width <= 0 or credit <= 0 or credit < width * MIN_CREDIT_FRAC:
        return None
    breakeven = short_leg.strike - d * credit
    return Structure(
        kind="bull_put_spread" if d > 0 else "bear_call_spread",
        label=(
            f"{short_leg.strike:g}/{long_leg.strike:g} "
            f"{'bull put' if d > 0 else 'bear call'} credit spread"
        ),
        unit="contract",
        legs=[_leg("sell", short_leg), _leg("buy", long_leg)],
        premium=-credit,
        max_loss=width - credit,
        max_gain=credit,
        breakeven=breakeven,
        pop_market_implied=pop_market_implied(
            spot=spot,
            level=breakeven,
            vol=_pop_vol([short_leg, long_leg], breakeven),
            t_years=years(short_leg.dte),
            profit_above=d > 0,
        ),
        rr=credit / (width - credit),
        premium_yield=credit / width,
    )


def iron_condor(
    chain: pd.DataFrame, band: Band, *, spot: float, asof: pd.Timestamp
) -> Structure | None:
    """30-45 DTE four-leg credit: short put strictly below the band's lower
    edge, short call strictly above its upper edge (the band is the
    invalidation, same semantics as credit_spread's stop), wings ~$5 out,
    total credit >= 1/3 of the widest wing."""
    exp = pick_expiration(chain, dte_lo=DTE_INCOME[0], dte_hi=DTE_INCOME[1], asof=asof)
    if exp is None:
        return None
    puts = liquid_quotes(chain, right="put", expiration=exp, spot=spot, asof=asof)
    calls = liquid_quotes(chain, right="call", expiration=exp, spot=spot, asof=asof)
    short_put = strictly_below(puts, band.lower)
    short_call = strictly_above(calls, band.upper)
    if short_put is None or short_call is None:
        return None
    long_put = nearest_strike(
        [q for q in puts if q.strike < short_put.strike], short_put.strike - SPREAD_WIDTH
    )
    long_call = nearest_strike(
        [q for q in calls if q.strike > short_call.strike], short_call.strike + SPREAD_WIDTH
    )
    if long_put is None or long_call is None:
        return None
    credit = short_put.mid - long_put.mid + short_call.mid - long_call.mid
    width = max(short_put.strike - long_put.strike, long_call.strike - short_call.strike)
    if width <= 0 or credit <= 0 or credit < width * MIN_CREDIT_FRAC:
        return None
    be_lo = short_put.strike - credit
    be_hi = short_call.strike + credit
    return Structure(
        kind="iron_condor",
        label=(
            f"{long_put.strike:g}/{short_put.strike:g}/"
            f"{short_call.strike:g}/{long_call.strike:g} iron condor"
        ),
        unit="contract",
        legs=[
            _leg("buy", long_put),
            _leg("sell", short_put),
            _leg("sell", short_call),
            _leg("buy", long_call),
        ],
        premium=-credit,
        max_loss=width - credit,
        max_gain=credit,
        breakeven=None,
        breakevens=[be_lo, be_hi],
        pop_market_implied=pop_range_market_implied(
            spot=spot,
            lower=be_lo,
            upper=be_hi,
            vol_lower=_pop_vol([short_put, long_put], be_lo),
            vol_upper=_pop_vol([short_call, long_call], be_hi),
            t_years=years(short_put.dte),
        ),
        rr=credit / (width - credit),
        premium_yield=credit / width,
    )


def iron_butterfly(
    chain: pd.DataFrame, band: Band, *, spot: float, asof: pd.Timestamp
) -> Structure | None:
    """30-45 DTE short straddle at the strike nearest spot, long wings at the
    strikes nearest the band edges. Tighter profit zone than the condor for
    more credit; breakevens are body +/- credit."""
    exp = pick_expiration(chain, dte_lo=DTE_INCOME[0], dte_hi=DTE_INCOME[1], asof=asof)
    if exp is None:
        return None
    puts = liquid_quotes(chain, right="put", expiration=exp, spot=spot, asof=asof)
    calls = liquid_quotes(chain, right="call", expiration=exp, spot=spot, asof=asof)
    body_call = nearest_strike(calls, spot)
    if body_call is None:
        return None
    body = body_call.strike
    body_put = next((q for q in puts if q.strike == body), None)
    if body_put is None:
        return None
    long_put = nearest_strike([q for q in puts if q.strike < body], band.lower)
    long_call = nearest_strike([q for q in calls if q.strike > body], band.upper)
    if long_put is None or long_call is None:
        return None
    credit = body_put.mid + body_call.mid - long_put.mid - long_call.mid
    width = max(body - long_put.strike, long_call.strike - body)
    if width <= 0 or credit <= 0 or width - credit <= 0:
        return None
    be_lo = body - credit
    be_hi = body + credit
    return Structure(
        kind="iron_butterfly",
        label=f"{long_put.strike:g}/{body:g}/{long_call.strike:g} iron butterfly",
        unit="contract",
        legs=[
            _leg("buy", long_put),
            _leg("sell", body_put),
            _leg("sell", body_call),
            _leg("buy", long_call),
        ],
        premium=-credit,
        max_loss=width - credit,
        max_gain=credit,
        breakeven=None,
        breakevens=[be_lo, be_hi],
        pop_market_implied=pop_range_market_implied(
            spot=spot,
            lower=be_lo,
            upper=be_hi,
            vol_lower=_pop_vol([body_put, long_put], be_lo),
            vol_upper=_pop_vol([body_call, long_call], be_hi),
            t_years=years(body_call.dte),
        ),
        rr=credit / (width - credit),
        premium_yield=credit / width,
    )


def build_neutral_structures(
    chain: pd.DataFrame, band: Band, *, spot: float, asof: pd.Timestamp
) -> list[Structure]:
    """Defined-risk range structures, condor first (it is recommended when both
    size). No stock plan — the neutral stance exists only on the chat path."""
    candidates = [
        iron_condor(chain, band, spot=spot, asof=asof),
        iron_butterfly(chain, band, spot=spot, asof=asof),
    ]
    return [s for s in candidates if s is not None]


def build_structures(
    chain: pd.DataFrame,
    levels: Levels,
    stance: str,
    *,
    spot: float,
    asof: pd.Timestamp,
    include_stock: bool = True,
) -> list[Structure]:
    """Everything that passes the gate, stock plan first — unless
    ``include_stock=False`` (the chat path is options-only). Naked short calls
    are never built (no builder exists for them — spec exclusion)."""
    out = [stock_plan(levels, stance)] if include_stock else []
    candidates = [long_option(chain, levels, spot=spot, asof=asof, stance=stance)]
    if direction(stance) > 0:
        candidates.append(cash_secured_put(chain, levels, spot=spot, asof=asof))
    candidates.append(credit_spread(chain, levels, spot=spot, asof=asof, stance=stance))
    out.extend(s for s in candidates if s is not None)
    return out
