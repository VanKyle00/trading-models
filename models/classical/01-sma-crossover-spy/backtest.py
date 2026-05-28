"""SMA crossover on SPY — the hello-world strategy for this repo.

Buy SPY when its 50-day SMA is above the 200-day SMA; flat otherwise.
This is *not* a strategy you'd deploy — it's a baseline that exercises the
full pipeline (data load → signal → backtest → metrics → plot) end-to-end
and proves the infrastructure works.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest import run_backtest
from tradinglib.loaders.equities.yfinance import load_daily

SYMBOL = "SPY"
FAST_WINDOW = 50
SLOW_WINDOW = 200
START = "2010-01-01"
END = "2024-12-31"


def build_signal(prices: pd.Series, fast: int = FAST_WINDOW, slow: int = SLOW_WINDOW) -> pd.Series:
    """Long when fast SMA > slow SMA, flat otherwise."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_daily(SYMBOL, start=START, end=END)
    prices = bars["close"]
    signal = build_signal(prices)

    result = run_backtest(prices, signal, fee_bps=1.0, slippage_bps=0.5)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(f"Wrote {metrics_path}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax, label=f"SMA {FAST_WINDOW}/{SLOW_WINDOW} crossover")
    buy_hold = (1.0 + prices.pct_change().fillna(0.0)).cumprod() * result.config["initial_capital"]
    buy_hold.plot(ax=ax, label="Buy & hold", alpha=0.6)
    ax.set_title(f"{SYMBOL} — SMA({FAST_WINDOW}/{SLOW_WINDOW}) crossover vs buy & hold")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = out_dir / "equity_curve.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
