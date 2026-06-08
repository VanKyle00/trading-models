"""EarningsStraddle: long ATM straddle around one earnings event.

Implements the OptionsStrategy protocol. Enters at T-entry_lead trading bars,
opens an ATM straddle expiring strictly after earnings, and closes at
T+exit_offset. The engine charges the bid/ask spread per leg, so a straddle pays
it twice on entry and twice on exit.

Phase-1 simplification (Component 2): expiry is approximated as
``earnings + post_earnings_tenor`` calendar days rather than snapped to a listed
weekly Friday; the listed-expiry snap arrives with the real chain in Phase 2/3.
``post_earnings_tenor`` defaults to 14 calendar days so the leg's expiry stays
comfortably beyond the (entry_lead + exit_offset) hold window — otherwise the
engine's _settle_expiries (run at the start of every bar) could settle the leg
before the strategy's exit bar. The exit branch keys off this strategy's own
entered/exited state, not engine.position.legs, so an already-settled leg never
breaks the exit accounting. Session label (bmo/amc) is parsed by the loader but
NOT used to offset entry/exit in this synthetic phase; the earnings bar is the
first bar whose timestamp is >= the earnings datetime (documented deviation).
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay

from tradinglib.backtest.options_engine import OptionsEngine
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
    ) -> None:
        self.earnings_datetime = _to_naive(earnings_datetime)
        self.entry_lead = entry_lead
        self.exit_offset = exit_offset
        self.contracts = contracts
        self.strike_step = strike_step
        self.post_earnings_tenor = post_earnings_tenor
        self._bars: list[pd.Timestamp] = []
        self.entered_on: pd.Timestamp | None = None
        self.exited_on: pd.Timestamp | None = None

    def _entry_target(self) -> pd.Timestamp:
        return self.earnings_datetime - BDay(self.entry_lead)

    def _exit_target(self) -> pd.Timestamp:
        return self.earnings_datetime + BDay(self.exit_offset)

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        self._bars.append(t)
        now = _to_naive(t)

        if self.entered_on is None and now >= self._entry_target() and now < self.earnings_datetime:
            expiry = self.earnings_datetime + pd.Timedelta(days=self.post_earnings_tenor)
            for leg in atm_straddle_legs(
                spot, expiry, quantity=self.contracts, strike_step=self.strike_step
            ):
                engine.add_leg(leg)
            self.entered_on = t
        elif self.entered_on is not None and self.exited_on is None and now >= self._exit_target():
            if engine.position.legs:
                engine.close_all_options()
            self.exited_on = t
