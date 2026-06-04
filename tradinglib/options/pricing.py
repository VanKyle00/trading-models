"""Option pricing and Greeks.

European options use the closed-form Black-Scholes-Merton model; American
options use a Cox-Ross-Rubinstein binomial tree that checks early exercise at
each node. All functions are pure and accept scalar floats (vectorize with
``numpy.vectorize`` at the call site if needed).

Conventions
-----------
- ``t`` is time to expiry in YEARS. ``t <= 0`` returns intrinsic value.
- ``vol`` and ``rate`` are annualized, expressed as decimals (0.20 = 20%).
- ``div`` is a continuous dividend yield (annualized decimal), default 0.
- ``right`` is ``"call"`` or ``"put"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Right = Literal["call", "put"]


def _intrinsic(right: Right, spot: float, strike: float) -> float:
    if right == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _d1_d2(spot: float, strike: float, t: float, vol: float, rate: float, div: float) -> tuple[float, float]:
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return d1, d2


def bs_price(
    right: Right,
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    div: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European option."""
    if t <= 0 or vol <= 0:
        return _intrinsic(right, spot, strike)
    d1, d2 = _d1_d2(spot, strike, t, vol, rate, div)
    disc = math.exp(-rate * t)
    carry = math.exp(-div * t)
    if right == "call":
        return spot * carry * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    return strike * disc * norm.cdf(-d2) - spot * carry * norm.cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    """First-order Greeks. ``vega``/``rho`` are per 1.0 of vol/rate (per 100%).

    Divide by 100 for per-1%-move conventions. ``theta`` is per YEAR; divide by
    365 for per-calendar-day.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bs_greeks(
    right: Right,
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    div: float = 0.0,
) -> Greeks:
    """First-order Black-Scholes Greeks for a European option."""
    if t <= 0 or vol <= 0:
        # At/after expiry: delta is a step function, other Greeks vanish.
        if right == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(spot, strike, t, vol, rate, div)
    pdf = norm.pdf(d1)
    disc = math.exp(-rate * t)
    carry = math.exp(-div * t)
    sqrt_t = math.sqrt(t)

    gamma = carry * pdf / (spot * vol * sqrt_t)
    vega = spot * carry * pdf * sqrt_t
    if right == "call":
        delta = carry * norm.cdf(d1)
        theta = (
            -spot * carry * pdf * vol / (2 * sqrt_t)
            - rate * strike * disc * norm.cdf(d2)
            + div * spot * carry * norm.cdf(d1)
        )
        rho = strike * t * disc * norm.cdf(d2)
    else:
        delta = -carry * norm.cdf(-d1)
        theta = (
            -spot * carry * pdf * vol / (2 * sqrt_t)
            + rate * strike * disc * norm.cdf(-d2)
            - div * spot * carry * norm.cdf(-d1)
        )
        rho = -strike * t * disc * norm.cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    price: float,
    right: Right,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div: float = 0.0,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Invert Black-Scholes for the volatility implied by an observed price.

    Uses Brent's method on ``[lo, hi]``. Raises ``ValueError`` if the price is
    below intrinsic (no real implied vol exists).
    """
    if price < _intrinsic(right, spot, strike) - 1e-9:
        raise ValueError(f"price {price} is below intrinsic value")

    def objective(vol: float) -> float:
        return bs_price(right, spot, strike, t, vol, rate, div) - price

    return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))


def crr_price(
    right: Right,
    spot: float,
    strike: float,
    t: float,
    vol: float,
    rate: float,
    style: Literal["european", "american"] = "american",
    div: float = 0.0,
    steps: int = 512,
) -> float:
    """Cox-Ross-Rubinstein binomial price. American style checks early exercise."""
    if t <= 0 or vol <= 0:
        return _intrinsic(right, spot, strike)

    dt = t / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rate * dt)
    p = (math.exp((rate - div) * dt) - d) / (u - d)

    # Terminal spot prices: spot * u^j * d^(steps-j) for j in 0..steps.
    j = np.arange(steps + 1)
    spot_t = spot * u**j * d ** (steps - j)
    if right == "call":
        values = np.maximum(spot_t - strike, 0.0)
    else:
        values = np.maximum(strike - spot_t, 0.0)

    # Backward induction.
    for step in range(steps, 0, -1):
        values = disc * (p * values[1:step + 1] + (1 - p) * values[0:step])
        if style == "american":
            j = np.arange(step)
            spot_nodes = spot * u**j * d ** (step - 1 - j)
            if right == "call":
                exercise = np.maximum(spot_nodes - strike, 0.0)
            else:
                exercise = np.maximum(strike - spot_nodes, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])
