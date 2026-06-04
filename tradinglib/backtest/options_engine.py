"""Event-driven options backtest engine.

Marks a multi-leg :class:`~tradinglib.options.instruments.Position` to market
over a price path (real or simulated). Unlike the vectorized engine, options
P&L is the *change in option value*, which is nonlinear in the underlying — so
this engine computes its own equity curve by mark-to-market rather than the
``position * return`` math in :func:`tradinglib.backtest.run_backtest`. It then
calls the shared :func:`compute_metrics` so results stay comparable.

``BacktestResult`` fields are reinterpreted for options (see
``docs/methodology.md``): ``position`` is net portfolio delta as a fraction of
equity; ``turnover`` is traded notional / equity.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from tradinglib.backtest.engine import BacktestResult
from tradinglib.backtest.metrics import compute_metrics
from tradinglib.options.instruments import CONTRACT_MULTIPLIER, OptionLeg, Position, intrinsic_value
from tradinglib.options.pricing import bs_greeks, bs_price, crr_price


class OptionsStrategy(Protocol):
    """Strategies implement a single per-bar callback."""

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None: ...


def _years_between(now: pd.Timestamp, expiry: pd.Timestamp) -> float:
    return max((expiry - now).days, 0) / 365.0


class OptionsEngine:
    """Per-bar dispatcher. The strategy adjusts the position via these methods;
    the engine handles pricing, cash accounting, costs, and turnover."""

    def __init__(
        self,
        vol: float,
        rate: float,
        fee_bps: float,
        slippage_bps: float,
        initial_capital: float,
    ) -> None:
        self.vol = vol
        self.rate = rate
        self.cost_rate = (fee_bps + slippage_bps) / 10_000.0
        self.position = Position(cash=initial_capital)
        self.spot: float = 0.0
        self.t: pd.Timestamp | None = None
        self._bar_notional: float = 0.0

    # --- pricing helpers ----------------------------------------------------
    def _price_leg(self, leg: OptionLeg) -> float:
        assert self.t is not None
        t_yrs = _years_between(self.t, leg.expiry)
        if t_yrs <= 0:
            return intrinsic_value(leg, self.spot)
        if leg.style == "american":
            return crr_price(leg.right, self.spot, leg.strike, t_yrs, self.vol, self.rate, style="american")
        return bs_price(leg.right, self.spot, leg.strike, t_yrs, self.vol, self.rate)

    def _leg_delta(self, leg: OptionLeg) -> float:
        assert self.t is not None
        t_yrs = _years_between(self.t, leg.expiry)
        return bs_greeks(leg.right, self.spot, leg.strike, t_yrs, self.vol, self.rate).delta

    def option_value(self) -> float:
        return sum(self._price_leg(leg) * leg.quantity * CONTRACT_MULTIPLIER for leg in self.position.legs)

    def net_delta_shares(self) -> float:
        """Net delta expressed in shares of the underlying (incl. hedge shares)."""
        opt = sum(self._leg_delta(leg) * leg.quantity * CONTRACT_MULTIPLIER for leg in self.position.legs)
        return opt + self.position.shares

    def equity(self) -> float:
        return self.position.cash + self.position.shares * self.spot + self.option_value()

    # --- trading API (called by strategies) ---------------------------------
    def add_leg(self, leg: OptionLeg) -> None:
        """Open (buy/sell) an option leg at its current model price."""
        price = self._price_leg(leg)
        notional = abs(leg.quantity) * price * CONTRACT_MULTIPLIER
        self.position.cash -= leg.quantity * price * CONTRACT_MULTIPLIER  # buy lowers cash
        self.position.cash -= notional * self.cost_rate
        self._bar_notional += notional
        self.position.legs.append(leg)

    def close_all_options(self) -> None:
        """Close every option leg at current model price."""
        for leg in self.position.legs:
            price = self._price_leg(leg)
            notional = abs(leg.quantity) * price * CONTRACT_MULTIPLIER
            self.position.cash += leg.quantity * price * CONTRACT_MULTIPLIER  # sell raises cash
            self.position.cash -= notional * self.cost_rate
            self._bar_notional += notional
        self.position.legs = []

    def hedge_to_delta(self, target_share_delta: float = 0.0) -> None:
        """Trade underlying shares so net share-delta equals the target."""
        delta_shares = target_share_delta - self.net_delta_shares()
        if delta_shares == 0.0:
            return
        notional = abs(delta_shares) * self.spot
        self.position.cash -= delta_shares * self.spot
        self.position.cash -= notional * self.cost_rate
        self._bar_notional += notional
        self.position.shares += delta_shares

    # --- internal -----------------------------------------------------------
    def _settle_expiries(self) -> None:
        assert self.t is not None
        survivors: list[OptionLeg] = []
        for leg in self.position.legs:
            if (leg.expiry - self.t).days <= 0:
                self.position.cash += leg.quantity * intrinsic_value(leg, self.spot) * CONTRACT_MULTIPLIER
            else:
                survivors.append(leg)
        self.position.legs = survivors


def run_options_backtest(
    prices: pd.Series,
    strategy: OptionsStrategy,
    *,
    vol: float,
    rate: float = 0.04,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run an options strategy over a price path and return a BacktestResult."""
    if len(prices) < 2:
        raise ValueError("need at least 2 bars to compute a return")

    engine = OptionsEngine(vol, rate, fee_bps, slippage_bps, initial_capital)
    equities: list[float] = []
    deltas: list[float] = []
    turnovers: list[float] = []

    for t, spot in prices.items():
        engine.t = pd.Timestamp(t)
        engine.spot = float(spot)
        engine._settle_expiries()
        engine._bar_notional = 0.0
        strategy.on_bar(engine, engine.t, engine.spot)

        eq = engine.equity()
        equities.append(eq)
        deltas.append(engine.net_delta_shares() * engine.spot / eq if eq != 0 else 0.0)
        turnovers.append(engine._bar_notional / eq if eq != 0 else 0.0)

    equity_curve = pd.Series(equities, index=prices.index, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)
    position = pd.Series(deltas, index=prices.index, name="delta_fraction")
    turnover = pd.Series(turnovers, index=prices.index, name="turnover")
    metrics = compute_metrics(returns, equity_curve, periods_per_year=periods_per_year)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        position=position,
        turnover=turnover,
        metrics=metrics,
        config={
            "initial_capital": initial_capital,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "periods_per_year": periods_per_year,
            "vol": vol,
            "rate": rate,
        },
    )
