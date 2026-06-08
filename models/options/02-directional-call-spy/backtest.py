"""Directional long call on SPY — naive vs realistic frictions.

Buys a ~2-month call (ATM or a configurable OTM offset), holds and rolls at
expiry. Runs the same strategy three ways — naive (flat vol, no spread), the
synthetic realized-vol surface without spread, and the surface plus a bid/ask
spread — so the headline is the P&L gap: the cost of paying real skew and spread
on a 2w-6mo option trade.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest.options_engine import OptionsEngine, run_options_backtest
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.options.instruments import OptionLeg
from tradinglib.options.spread import NoSpread, ParametricSpread
from tradinglib.options.surface import FlatSurface, realistic_surface

SYMBOL = "SPY"
START = "2023-01-01"
END = "2024-12-31"
TENOR_DAYS = 60
OTM_PCT = 0.0
FLAT_VOL = 0.18


class DirectionalCall:
    """Hold one long call; open at the configured OTM offset, roll at expiry."""

    def __init__(self, tenor_days: int = TENOR_DAYS, otm_pct: float = OTM_PCT) -> None:
        self.tenor_days = tenor_days
        self.otm_pct = otm_pct

    def on_bar(self, engine: OptionsEngine, t: pd.Timestamp, spot: float) -> None:
        if not engine.position.legs:
            strike = float(round(spot * (1.0 + self.otm_pct)))
            expiry = t + pd.Timedelta(days=self.tenor_days)
            engine.add_leg(OptionLeg("call", strike=strike, expiry=expiry, quantity=1.0))


def run_compare(
    prices: pd.Series, *, tenor_days: int = TENOR_DAYS, otm_pct: float = OTM_PCT
) -> dict:
    """Run the strategy three ways and return the BacktestResults.

    - ``naive_flat``: the optimistic baseline (constant vol, no spread).
    - ``surface_no_spread``: realistic vol surface, but no spread.
    - ``surface_with_spread``: realistic surface + bid/ask spread.

    Comparing the last two isolates the spread cost (surface held fixed);
    comparing ``naive_flat`` against ``surface_with_spread`` is the full
    optimistic-vs-realistic headline.
    """
    surface = realistic_surface(prices)
    naive_flat = run_options_backtest(
        prices,
        DirectionalCall(tenor_days, otm_pct),
        surface=FlatSurface(FLAT_VOL),
        spread=NoSpread(),
    )
    surface_no_spread = run_options_backtest(
        prices,
        DirectionalCall(tenor_days, otm_pct),
        surface=surface,
        spread=NoSpread(),
    )
    surface_with_spread = run_options_backtest(
        prices,
        DirectionalCall(tenor_days, otm_pct),
        surface=surface,
        spread=ParametricSpread(),
    )
    return {
        "naive_flat": naive_flat,
        "surface_no_spread": surface_no_spread,
        "surface_with_spread": surface_with_spread,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_daily(SYMBOL, start=START, end=END)["close"]
    out = run_compare(prices)

    summary = {
        key: {
            "metrics": res.metrics,
            "final_equity": float(res.equity_curve.iloc[-1]),
        }
        for key, res in out.items()
    }
    (out_dir / "compare.json").write_text(json.dumps(summary, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(10, 5))
    out["naive_flat"].equity_curve.plot(ax=ax, label="Naive (flat vol, no spread)")
    out["surface_no_spread"].equity_curve.plot(ax=ax, label="Surface, no spread")
    out["surface_with_spread"].equity_curve.plot(ax=ax, label="Surface + spread (realistic)")
    ax.set_title(f"{SYMBOL} — directional call, naive vs realistic frictions")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "compare_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
