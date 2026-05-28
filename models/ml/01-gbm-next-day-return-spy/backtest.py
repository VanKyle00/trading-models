"""Backtest the trained XGBoost model on the held-out OOS window.

Loads the artifact written by ``train.py``, predicts next-day log returns
on the OOS window, converts predictions into a position (long when the
predicted return is positive, flat otherwise), and runs the standard
backtest engine. Saves ``results/metrics.json`` and ``results/equity_curve.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import END, START, SYMBOL, build_features  # noqa: E402

from tradinglib.backtest import run_backtest  # noqa: E402
from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402

RESULTS = HERE / "results"


def main() -> None:
    bundle = joblib.load(RESULTS / "model.joblib")
    model = bundle["model"]
    features = bundle["features"]
    split = bundle["split_index"]

    bars = load_daily(SYMBOL, start=START, end=END)
    prices = bars["close"]
    feats = build_features(prices)

    x_oos = feats.loc[feats.index >= split, features].dropna()
    pred = pd.Series(model.predict(x_oos), index=x_oos.index)

    # Long when predicted next-day log return is positive, flat otherwise
    signal = (pred > 0).astype(float)
    oos_prices = prices.loc[signal.index]

    result = run_backtest(oos_prices, signal, fee_bps=1.0, slippage_bps=0.5)

    metrics_path = RESULTS / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(f"Wrote {metrics_path}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax, label="XGBoost (OOS)")
    buy_hold = (1.0 + oos_prices.pct_change().fillna(0.0)).cumprod() * result.config[
        "initial_capital"
    ]
    buy_hold.plot(ax=ax, label="Buy & hold (OOS)", alpha=0.6)
    ax.set_title(f"{SYMBOL} — XGBoost next-day return model (OOS)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = RESULTS / "equity_curve.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
