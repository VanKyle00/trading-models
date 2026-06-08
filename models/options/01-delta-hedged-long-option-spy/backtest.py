"""Delta-hedged long option on SPY — the options 'hello-world'.

Buy a ~1-month ATM call, delta-hedge to zero with the underlying every bar,
and roll a fresh option whenever the previous one expires. Because the
position is continuously delta-hedged, its P&L isolates the gap between
*realized* volatility (how much SPY actually moved) and the *implied*
volatility we priced the option at — the cleanest exercise of the pricing +
Greeks machinery.

European exercise is used for the clean vol story; the American CRR pricer is
validated separately in the test suite.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.options.instruments import CONTRACT_MULTIPLIER, OptionLeg
from tradinglib.options.pricing import Right, bs_price
from tradinglib.options.simulate import run_simulation
from tradinglib.options.surface import FlatSurface

SYMBOL = "SPY"
START = "2023-01-01"
END = "2024-12-31"
TENOR_DAYS = 30
IMPLIED_VOL = 0.18
RATE = 0.04
FEE_BPS = 1.0
SLIPPAGE_BPS = 0.5


class DeltaHedgedLongOption:
    """Hold one long ATM call, delta-hedge to zero each bar, roll at expiry."""

    def __init__(self, tenor_days: int = TENOR_DAYS, right: Right = "call") -> None:
        self.tenor_days = tenor_days
        self.right = right

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        if not engine.position.legs:
            expiry = t + pd.Timedelta(days=self.tenor_days)
            engine.add_leg(
                OptionLeg(self.right, strike=float(round(spot)), expiry=expiry, quantity=1.0)
            )
        engine.hedge_to_delta(0.0)


def _payoff_curve(
    spot: float, strike: float, vol: float, rate: float, tenor_days: int
) -> dict[str, Any]:
    """Value of one long call vs spot, today and at expiry — for the GUI plot."""
    spots = np.linspace(spot * 0.8, spot * 1.2, 80)
    t_yrs = tenor_days / 365.0
    today = np.array(
        [bs_price("call", s, strike, t_yrs, vol, rate) * CONTRACT_MULTIPLIER for s in spots]
    )
    at_expiry = np.maximum(spots - strike, 0.0) * CONTRACT_MULTIPLIER
    return {"spots": spots, "values": today, "expiry_values": at_expiry, "strike": strike}


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    implied_vol: float = IMPLIED_VOL,
    tenor_days: int = TENOR_DAYS,
    n_paths: int = 2_000,
) -> dict[str, Any]:
    """Run the historical backtest + Monte Carlo simulation without writing to disk.

    Returns a dict with:
      - ``data``: DataFrame indexed by timestamp with ``close`` and
        ``delta_fraction`` columns — used by the GUI's data view.
      - ``result``: the :class:`BacktestResult` from ``run_options_backtest``.
      - ``simulation``: :class:`SimulationResult` from Monte Carlo.
      - ``payoff``: dict with spots/values/expiry_values/strike for the GUI plot.
      - ``symbol``: the traded symbol, for plot titles.
      - ``params``: dict echo of the inputs.
    """
    bars = load_daily(symbol, start=str(start), end=str(end))
    prices = bars["close"]

    result = run_options_backtest(
        prices,
        DeltaHedgedLongOption(tenor_days=tenor_days),
        surface=FlatSurface(implied_vol),
        rate=RATE,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )

    def factory() -> DeltaHedgedLongOption:
        return DeltaHedgedLongOption(tenor_days=tenor_days)

    spot0 = float(prices.iloc[0])
    simulation = run_simulation(
        factory,
        spot=spot0,
        vol=implied_vol,
        rate=RATE,
        # Simulate a fixed horizon of `tenor_days` business-day bars (capped by the
        # historical window length). The option's expiry is tenor_days CALENDAR days
        # out (~21 business days for 30), so within the path it expires, settles to
        # intrinsic, and a fresh option is opened — the MC reflects the same rolling,
        # delta-hedged strategy as the historical backtest, not a single option held
        # to expiry.
        days=min(tenor_days, len(prices)),
        n_paths=n_paths,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        seed=0,
    )

    data = pd.DataFrame({"close": prices, "delta_fraction": result.position})
    payoff = _payoff_curve(spot0, round(spot0), implied_vol, RATE, tenor_days)
    return {
        "data": data,
        "result": result,
        "simulation": simulation,
        "payoff": payoff,
        "symbol": symbol,
        "params": {
            "start": str(start),
            "end": str(end),
            "symbol": symbol,
            "implied_vol": implied_vol,
            "tenor_days": tenor_days,
            "n_paths": n_paths,
        },
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = run_for_gui()
    result = out["result"]

    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Delta-hedged option")
    ax.set_title("Delta-hedged long option on SPY")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.savefig(out_dir / "equity_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
