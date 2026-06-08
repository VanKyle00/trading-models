"""ATM straddle assembly over existing option primitives.

No new pricing: a straddle is a long call + long put at the nearest listed
strike, same expiry. Strikes snap to a configurable ``strike_step`` grid to
approximate the listed chain in the Phase-1 synthetic pipeline.
"""

from __future__ import annotations

import pandas as pd

from tradinglib.options.instruments import OptionLeg
from tradinglib.options.pricing import bs_price


def snap_strike(spot: float, strike_step: float) -> float:
    """Round ``spot`` to the nearest multiple of ``strike_step``."""
    if strike_step <= 0:
        raise ValueError(f"strike_step must be > 0, got {strike_step}")
    return float(round(spot / strike_step) * strike_step)


def atm_straddle_legs(
    spot: float,
    expiry: pd.Timestamp,
    quantity: float = 1.0,
    *,
    underlying: str = "SPY",
    strike_step: float = 1.0,
) -> list[OptionLeg]:
    """Return ``[call_leg, put_leg]`` for a long ATM straddle."""
    strike = snap_strike(spot, strike_step)
    return [
        OptionLeg("call", strike=strike, expiry=expiry, quantity=quantity, underlying=underlying),
        OptionLeg("put", strike=strike, expiry=expiry, quantity=quantity, underlying=underlying),
    ]


def straddle_price(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    div: float = 0.0,
) -> float:
    """Synthetic ATM-straddle premium = call BS + put BS (per share)."""
    call = bs_price("call", spot, strike, t, vol, rate, div)
    put = bs_price("put", spot, strike, t, vol, rate, div)
    return call + put
