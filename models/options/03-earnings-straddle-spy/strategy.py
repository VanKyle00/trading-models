"""EarningsStraddle: long ATM straddle around one earnings event.

Implements the OptionsStrategy protocol. Enters at T-entry_lead trading bars,
opens an ATM straddle expiring strictly after earnings, and closes at
T+exit_offset. The engine charges the bid/ask spread per leg, so a straddle pays
it twice on entry and twice on exit.

Timing counts ACTUAL price bars, not calendar business days: the earnings bar is
the first bar whose timestamp is >= the earnings datetime, and entry/exit are the
``entry_lead``-th bar before / ``exit_offset``-th bar after that bar in the price
index. Counting actual bars (rather than ``earnings ± BDay(n)``) keeps the timing
correct on gapped/holiday calendars, where a business-day offset can land on a
date with no bar. Because the entry bar precedes the earnings bar, the strategy
needs the bar schedule up front: ``bar_index`` (the backtest's price index) is
required so entry/exit indices are computed deterministically. Knowing the
trading calendar is not price lookahead.

Phase-1 simplification (Component 2): expiry is approximated as
``earnings + post_earnings_tenor`` calendar days rather than snapped to a listed
weekly Friday; the listed-expiry snap arrives with the real chain in Phase 2/3.
``post_earnings_tenor`` defaults to 14 calendar days so the leg's expiry stays
comfortably beyond the (entry_lead + exit_offset) hold window — otherwise the
engine's _settle_expiries (run at the start of every bar) could settle the leg
before the strategy's exit bar. The exit branch keys off this strategy's own
entered/exited state, not engine.position.legs, so an already-settled leg never
breaks the exit accounting. Session label (bmo/amc) is parsed by the loader but
NOT used to offset entry/exit in this synthetic phase (documented deviation).
"""

from __future__ import annotations

import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine
from tradinglib.options.instruments import CONTRACT_MULTIPLIER
from tradinglib.options.straddle import atm_straddle_legs


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


class EarningsStraddle:
    def __init__(
        self,
        *,
        earnings_datetime: pd.Timestamp,
        entry_lead: int = 3,
        exit_offset: int = 1,
        contracts: float = 1.0,
        strike_step: float = 1.0,
        post_earnings_tenor: int = 14,
        bar_index: pd.DatetimeIndex,
    ) -> None:
        self.earnings_datetime = _to_naive(earnings_datetime)
        self.entry_lead = entry_lead
        self.exit_offset = exit_offset
        self.contracts = contracts
        self.strike_step = strike_step
        self.post_earnings_tenor = post_earnings_tenor
        self.entered_on: pd.Timestamp | None = None
        self.exited_on: pd.Timestamp | None = None
        # Exact entry/exit timestamps planned from the bar schedule.
        self._entry_on: pd.Timestamp | None = None
        self._exit_on: pd.Timestamp | None = None
        self._plan_from_index(bar_index)

    def _plan_from_index(self, bar_index: pd.DatetimeIndex) -> None:
        naive = pd.DatetimeIndex([_to_naive(b) for b in bar_index])
        after = naive >= self.earnings_datetime
        if not after.any():
            return  # earnings beyond the data window; no trade
        e_idx = int(after.argmax())  # first bar at/after earnings
        entry_idx = e_idx - self.entry_lead
        exit_idx = e_idx + self.exit_offset
        if 0 <= entry_idx < len(bar_index):
            self._entry_on = _to_naive(bar_index[entry_idx])
        if 0 <= exit_idx < len(bar_index):
            self._exit_on = _to_naive(bar_index[exit_idx])

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        now = _to_naive(t)

        # Entry: the exact bar planned from the schedule.
        if self.entered_on is None and self._entry_on is not None and now == self._entry_on:
            expiry = self.earnings_datetime + pd.Timedelta(days=self.post_earnings_tenor)
            for leg in atm_straddle_legs(
                spot, expiry, quantity=self.contracts, strike_step=self.strike_step
            ):
                engine.add_leg(leg)
            self.entered_on = t
            return

        # Exit: the exact bar planned from the schedule.
        if (
            self.entered_on is not None
            and self.exited_on is None
            and self._exit_on is not None
            and now >= self._exit_on
        ):
            if engine.position.legs:
                engine.close_all_options()
            self.exited_on = t


def size_contracts(capital: float, risk_fraction: float, straddle_premium: float) -> float:
    """Fixed-fractional sizing: contracts = floor(risk_budget / cost_per_contract).

    Long options are defined-risk (max loss = premium paid), so the risk budget
    is ``capital * risk_fraction`` and per-contract cost is
    ``straddle_premium * CONTRACT_MULTIPLIER``.
    """
    if straddle_premium <= 0:
        return 0.0
    budget = capital * risk_fraction
    cost_per_contract = straddle_premium * CONTRACT_MULTIPLIER
    return float(int(budget // cost_per_contract))


def can_open(open_count: int, max_concurrent: int) -> bool:
    """Portfolio cap: True iff fewer than ``max_concurrent`` straddles are open."""
    return open_count < max_concurrent
