"""Multi-leg options position model.

An :class:`OptionLeg` is one contract line (right, strike, expiry, signed
quantity). A :class:`Position` is the unit the backtest engine marks to market:
a list of legs plus underlying ``shares`` and ``cash``. The standard equity
contract multiplier is 100 (one contract controls 100 shares).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from tradinglib.options.pricing import Right, intrinsic

CONTRACT_MULTIPLIER = 100.0

Style = Literal["european", "american"]


@dataclass(frozen=True)
class OptionLeg:
    """One option contract line. ``quantity`` is signed: +long, -short."""

    right: Right
    strike: float
    expiry: pd.Timestamp
    quantity: float
    style: Style = "european"
    underlying: str = "SPY"


def intrinsic_value(leg: OptionLeg, spot: float) -> float:
    """Per-contract intrinsic value of one leg's option (unsigned by quantity)."""
    return intrinsic(leg.right, spot, leg.strike)


@dataclass
class Position:
    """A multi-leg options position plus underlying shares and cash."""

    legs: list[OptionLeg] = field(default_factory=list)
    shares: float = 0.0
    cash: float = 0.0

    def intrinsic_value(self, spot: float) -> float:
        """Mark every leg at intrinsic value (used at/after expiry)."""
        options = sum(
            leg.quantity * intrinsic_value(leg, spot) * CONTRACT_MULTIPLIER for leg in self.legs
        )
        return options + self.shares * spot + self.cash
