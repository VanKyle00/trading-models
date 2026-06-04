"""SMA crossover on SPY — the hello-world strategy for this repo.

Buy SPY when its 50-day SMA is above the 200-day SMA; flat otherwise.
This is *not* a strategy you'd deploy — it's a baseline that exercises the
full pipeline (data load → signal → backtest → metrics → plot) end-to-end
and proves the infrastructure works.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest import run_backtest
from tradinglib.loaders.equities.yfinance import load_daily

SYMBOL = "SPY"
FAST_WINDOW = 50
SLOW_WINDOW = 200
START = "2010-01-01"
END = "2024-12-31"
FEE_BPS = 1.0
SLIPPAGE_BPS = 0.5


def build_signal(prices: pd.Series, fast: int = FAST_WINDOW, slow: int = SLOW_WINDOW) -> pd.Series:
    """Long when fast SMA > slow SMA, flat otherwise."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
    fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    initial_capital: float = 100_000.0,
    size_mult: float = 1.0,
) -> dict[str, Any]:
    """Run the SMA crossover end-to-end without writing to disk.

    Returns a dict with:
      - ``data``: DataFrame indexed by timestamp with ``close``, ``fast_ma``,
        ``slow_ma``, and ``signal`` columns — used by the GUI's data view.
      - ``result``: the :class:`BacktestResult` from ``run_backtest``.
      - ``symbol``: the traded symbol, for plot titles.
      - ``params``: dict echo of the inputs.
    """
    bars = load_daily(symbol, start=start, end=end)
    prices = bars["close"]
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float) * size_mult

    result = run_backtest(
        prices,
        signal,
        execution_prices=bars["open"],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
    )

    data = pd.DataFrame({"close": prices, "fast_ma": fast_ma, "slow_ma": slow_ma, "signal": signal})
    return {
        "data": data,
        "result": result,
        "symbol": symbol,
        "params": {
            "start": str(start),
            "end": str(end),
            "symbol": symbol,
            "fast": fast,
            "slow": slow,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "size_mult": size_mult,
        },
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = run_for_gui(START, END)
    result = out["result"]
    prices = out["data"]["close"]

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
