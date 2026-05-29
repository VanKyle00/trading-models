"""Order Flow Imbalance on BTCUSDT — microstructure seed model.

Pulls three days of BTCUSDT aggregated trades from Binance's public CDN
(default window brackets the 2024-08-05 crash), aggregates them into
1-minute bars with order-flow-imbalance (OFI) as a per-bar feature, and
trades a simple threshold rule via the event-driven backtest engine.

**Hypothesis.** Aggressive buying — taker orders lifting the ask —
generates short-term upward pressure on price; aggressive selling pushes
price down. If the recent average OFI is meaningfully positive, going
long for the next bar should capture some of that continuation;
meaningfully negative OFI flips the position short. This is the canonical
trade-flow / market-impact effect that microstructure research has
documented since at least Hasbrouck (1991).

**Data.** Three days of BTCUSDT aggregated trades from
``data.binance.vision`` covering the 2024-08-05 crash (pre-crash +
crash + recovery). The loader caches the parsed trades to parquet so
subsequent runs do not re-download.

**Signal.** Per minute, compute OFI as
``(buy_volume - sell_volume) / total_volume``. Smooth with a 5-minute
trailing average. Position rule:

- smoothed OFI > +0.20 → long
- smoothed OFI < -0.20 → short
- otherwise → flat

The event-driven engine processes bars sequentially and the strategy
maintains its own rolling buffer — natural fit for the imperative API.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest import bars_from_dataframe, run_event_backtest
from tradinglib.backtest.event_engine import Bar, EventEngine
from tradinglib.features.microstructure import aggregate_to_bars
from tradinglib.loaders.crypto.binance_trades import load_trades

SYMBOL = "BTCUSDT"
START = "2024-08-04"
END = "2024-08-06"
BAR_SECONDS = 60
SMOOTH_WINDOW = 5
ENTRY_THRESHOLD = 0.20
PERIODS_PER_YEAR = 525_600
FEE_BPS = 2.0
SLIPPAGE_BPS = 5.0
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


class RollingOFIStrategy:
    """Trades on the trailing average of order flow imbalance."""

    def __init__(
        self,
        ofi_lookup: dict[pd.Timestamp, float],
        window: int,
        threshold: float,
    ) -> None:
        self._lookup = ofi_lookup
        self._window = window
        self._threshold = threshold
        self._buf: list[float] = []

    def on_bar(self, engine: EventEngine, bar: Bar) -> None:
        val = self._lookup.get(bar.timestamp)
        if val is None or pd.isna(val):
            return

        self._buf.append(float(val))
        if len(self._buf) > self._window:
            self._buf.pop(0)
        if len(self._buf) < self._window:
            return

        avg = sum(self._buf) / self._window
        if avg > self._threshold:
            engine.set_target_position(1.0)
        elif avg < -self._threshold:
            engine.set_target_position(-1.0)
        else:
            engine.set_target_position(0.0)


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    bar_seconds: int = BAR_SECONDS,
    smooth_window: int = SMOOTH_WINDOW,
    entry_threshold: float = ENTRY_THRESHOLD,
) -> dict[str, Any]:
    """OFI backtest on a user-selected date range.

    First call downloads aggTrades ZIPs from Binance's public CDN and
    caches per-day parquet (~50-100 MB per BTCUSDT-day). Subsequent
    calls within the cached range are fast.
    """
    if symbol != SYMBOL:
        raise ValueError(
            f"this model is locked to {SYMBOL} (Binance aggTrades data); got {symbol!r}"
        )
    trades = load_trades(SYMBOL, start, end)
    bars_df = aggregate_to_bars(trades, bar_seconds=bar_seconds)

    if len(bars_df) < smooth_window + 2:
        raise ValueError(
            f"only {len(bars_df)} bars in [{start}, {end}] — need at least "
            f"{smooth_window + 2} for the smoothed OFI signal"
        )

    bars = bars_from_dataframe(bars_df[["open", "high", "low", "close", "volume"]])
    ofi_lookup = bars_df["ofi"].to_dict()

    strategy = RollingOFIStrategy(
        ofi_lookup=ofi_lookup,
        window=smooth_window,
        threshold=entry_threshold,
    )
    result = run_event_backtest(
        bars,
        strategy,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        periods_per_year=PERIODS_PER_YEAR,
    )

    smoothed = bars_df["ofi"].rolling(smooth_window).mean()
    data = pd.DataFrame(
        {
            "close": bars_df["close"],
            "ofi": bars_df["ofi"],
            "ofi_smoothed": smoothed,
        }
    )
    return {
        "data": data,
        "result": result,
        "symbol": SYMBOL,
        "params": {
            "start": str(start),
            "end": str(end),
            "bar_seconds": bar_seconds,
            "smooth_window": smooth_window,
            "entry_threshold": entry_threshold,
        },
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SYMBOL} trades {START} -> {END} ...")
    out = run_for_gui(START, END)
    result = out["result"]
    data = out["data"]

    metrics_path = RESULTS / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(f"Wrote {metrics_path}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    result.equity_curve.plot(ax=axes[0], label="OFI strategy")
    bh = (1.0 + data["close"].pct_change().fillna(0.0)).cumprod() * result.config["initial_capital"]
    bh.plot(ax=axes[0], label="Buy & hold (1-minute)", alpha=0.6)
    axes[0].set_title(f"{SYMBOL} {START} to {END} — rolling OFI vs buy & hold (1-minute bars)")
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    data["ofi_smoothed"].plot(ax=axes[1], color="steelblue", linewidth=0.7, alpha=0.8)
    axes[1].axhline(ENTRY_THRESHOLD, color="green", linestyle="--", linewidth=0.7, alpha=0.6)
    axes[1].axhline(-ENTRY_THRESHOLD, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
    axes[1].axhline(0.0, color="black", linewidth=0.3, alpha=0.3)
    axes[1].set_ylabel(f"OFI ({SMOOTH_WINDOW}-bar smoothed)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS / "equity_curve.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
