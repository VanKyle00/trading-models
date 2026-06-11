"""Deterministic exit plan for a chat ticket's recommended structure.

Every number is computed, never narrated: price triggers reprice the whole
structure with Black-Scholes at the user's own levels (each leg at its chain
IV, held constant) evaluated at the half-remaining-DTE date; thresholds are
standard management practice expressed as order prices for THIS ticket —
credit structures take profit at 50% of max gain and cut at a buyback of 2x
the credit, debit structures take profit at +100% of the debit (or the target
trigger, whichever hits first) and cut at -50%. Time rules: credit structures
close/roll at 21 DTE, debit structures get a time stop at the final third of
DTE. Unsized structures report 1-lot numbers (the calculator-link convention).
Pure; no I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from tradinglib.options.pricing import bs_price
from tradinglib.strategist.quotes import years
from tradinglib.strategist.structures import Band, Structure
from tradinglib.tournament.levels import Levels

CREDIT_TP_FRAC = 0.50  # take profit at 50% of max gain
CREDIT_CUT_MULT = 2.0  # close when the buyback cost reaches 2x the credit
DEBIT_TP_MULT = 2.0  # take profit at +100% of the debit
DEBIT_CUT_FRAC = 0.50  # cut at -50% of the debit
CREDIT_MANAGE_DTE = 21
DEBIT_TIME_STOP_FRAC = 1.0 / 3.0  # final third of DTE

SHARES_PER_CONTRACT = 100


@dataclass(frozen=True)
class PlanRule:
    trigger: str
    action: str
    est_value: float | None = None  # structure value per share at the trigger
    est_pnl: float | None = None  # dollars for the sized position (1-lot if unsized)
    est_pnl_pct: float | None = None  # vs debit (debit structures) or max gain (credit)


@dataclass(frozen=True)
class ExitPlan:
    price_rules: list[PlanRule]
    profit_take: PlanRule | None
    loss_cut: PlanRule | None
    time_rules: list[PlanRule]
    theta_week: float | None  # per share per 7 days, copied from the structure
    est_by: str | None  # price-rule estimates assume the level is hit by this date
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _value_per_share(s: Structure, *, spot: float, t: float) -> float:
    total = 0.0
    for leg in s.legs:
        sign = 1.0 if leg["action"] == "buy" else -1.0
        total += sign * bs_price(leg["right"], spot, leg["strike"], t, leg["iv"], 0.0)
    return total


def build_exit_plan(
    s: Structure,
    levels: Levels | Band,
    *,
    next_earnings: pd.Timestamp | None = None,
) -> ExitPlan | None:
    """The trade plan for one option structure; None for leg-less (stock) plans."""
    if not s.legs or s.premium is None:
        return None
    premium = s.premium
    mult = SHARES_PER_CONTRACT * (s.quantity or 1)
    dte = int(s.legs[0]["dte"])
    # every builder emits single-expiration structures; legs share one expiry
    expiration = pd.Timestamp(s.legs[0]["expiration"])
    half_left = dte - dte // 2
    est_by = (expiration - pd.Timedelta(days=half_left)).strftime("%Y-%m-%d")
    t_half = years(half_left)
    credit = -premium if premium < 0 else None
    basis = credit if credit is not None else premium  # pct vs max gain / vs debit

    def rule(trigger: str, action: str, spot_at: float) -> PlanRule:
        value = _value_per_share(s, spot=spot_at, t=t_half)
        pnl_share = value - premium
        return PlanRule(
            trigger=trigger,
            action=action,
            est_value=round(value, 2),
            est_pnl=round(pnl_share * mult, 0),
            est_pnl_pct=round(pnl_share / basis, 2) if basis else None,
        )

    if isinstance(levels, Band):
        price_rules = [
            rule(f"underlying <= {levels.lower:g} (band floor breached)", "close", levels.lower),
            rule(f"underlying >= {levels.upper:g} (band ceiling breached)", "close", levels.upper),
        ]
    else:
        price_rules = [
            rule(f"underlying reaches the target {levels.target:g}", "take profit", levels.target),
            rule(f"underlying reaches the stop {levels.stop:g}", "close", levels.stop),
        ]

    profit_take: PlanRule | None = None
    loss_cut: PlanRule | None = None
    time_rules: list[PlanRule] = []
    if credit is not None:
        profit_take = PlanRule(
            trigger=(
                f"structure can be bought back at <= ${(1.0 - CREDIT_TP_FRAC) * credit:.2f}/sh "
                f"({CREDIT_TP_FRAC:.0%} of max gain)"
            ),
            action="take profit",
            est_pnl=round(CREDIT_TP_FRAC * credit * mult, 0),
            est_pnl_pct=CREDIT_TP_FRAC,
        )
        loss_cut = PlanRule(
            trigger=(
                f"buyback cost reaches ${CREDIT_CUT_MULT * credit:.2f}/sh "
                f"({CREDIT_CUT_MULT:g}x the credit received)"
            ),
            action="close",
            est_pnl=round(-credit * mult, 0),
            est_pnl_pct=-1.0,
        )
        if dte > CREDIT_MANAGE_DTE:
            when = (expiration - pd.Timedelta(days=CREDIT_MANAGE_DTE)).strftime("%Y-%m-%d")
            time_rules.append(
                PlanRule(trigger=f"{CREDIT_MANAGE_DTE} DTE ({when})", action="close or roll")
            )
    elif premium > 0:
        profit_take = PlanRule(
            trigger=(
                f"structure marks at >= ${DEBIT_TP_MULT * premium:.2f}/sh "
                f"(+{DEBIT_TP_MULT - 1:.0%} of the debit) "
                "— or the target trigger, whichever hits first"
            ),
            action="take profit",
            est_pnl=round(premium * mult, 0),
            est_pnl_pct=1.0,
        )
        loss_cut = PlanRule(
            trigger=f"structure marks at <= ${DEBIT_CUT_FRAC * premium:.2f}/sh (-{DEBIT_CUT_FRAC:.0%} of the debit)",
            action="close",
            est_pnl=round(-DEBIT_CUT_FRAC * premium * mult, 0),
            est_pnl_pct=-DEBIT_CUT_FRAC,
        )
        days_left = max(1, int(dte * DEBIT_TIME_STOP_FRAC))
        when = (expiration - pd.Timedelta(days=days_left)).strftime("%Y-%m-%d")
        time_rules.append(
            PlanRule(
                trigger=f"no progress toward the target by {when} (final third of DTE)",
                action="exit — the thesis is stale",
            )
        )

    notes: list[str] = []
    if next_earnings is not None:
        ne = (
            next_earnings.tz_convert("UTC").tz_localize(None)
            if next_earnings.tzinfo
            else next_earnings
        )
        if ne.normalize() <= expiration:
            notes.append(
                f"earnings {ne:%Y-%m-%d} before expiry — decide beforehand: close or accept the gap"
            )

    return ExitPlan(
        price_rules=price_rules,
        profit_take=profit_take,
        loss_cut=loss_cut,
        time_rules=time_rules,
        theta_week=s.theta_week,
        est_by=est_by,
        notes=notes,
    )
