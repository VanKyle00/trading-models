"""Trade levels: a tournament winner's rule turned into tomorrow's numbers.

``Levels`` is the contract between the tournament (sub-project B) and the
ticket playbook (sub-project C): entry with its trigger type, protective
stop, target, and a one-line plain-English condition. Rules without a native
loss-side exit use a 2x ATR(14) stop; targets default to 2R unless the rule
has a natural exit level (e.g. the Bollinger mean).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from tradinglib.features.technical import atr

_ATR_WINDOW = 14
_ATR_MULT = 2.0


def direction(stance: str) -> int:
    """+1 for ``"long"``, -1 for ``"short"``."""
    if stance == "long":
        return 1
    if stance == "short":
        return -1
    raise ValueError(f"stance must be 'long' or 'short', got {stance!r}")


@dataclass(frozen=True)
class Levels:
    """Tomorrow's actionable numbers for one ticker/strategy/stance."""

    entry: float
    entry_type: str  # "market" | "stop" | "limit"
    stop: float
    target: float
    condition: str

    def as_dict(self) -> dict:
        return asdict(self)


def protective_stop(bars: pd.DataFrame, entry: float, stance: str) -> float:
    """Default stop for rules with no native loss-side exit: 2x ATR(14)."""
    atr_value = float(atr(bars["high"], bars["low"], bars["close"], _ATR_WINDOW).iloc[-1])
    if not atr_value > 0.0:
        raise ValueError(f"degenerate ATR ({atr_value}); cannot place a stop")
    return entry - direction(stance) * _ATR_MULT * atr_value


def two_r_target(entry: float, stop: float) -> float:
    """Default target: entry +/- 2R, where R is the entry-to-stop distance."""
    return entry + 2.0 * (entry - stop)
