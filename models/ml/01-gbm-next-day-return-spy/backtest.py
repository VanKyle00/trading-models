"""Backtest the trained XGBoost model on the held-out OOS window.

Loads the artifact written by ``train.py``, predicts next-day log returns
on the OOS window, converts predictions into a position (long when the
predicted return is positive, flat otherwise), and runs the standard
backtest engine. Saves ``results/metrics.json`` and ``results/equity_curve.png``.

Exposes :func:`run_for_gui` so the Streamlit app can predict + backtest
on user-selected date windows without re-training.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import END, START, SYMBOL, build_features, train_model  # noqa: E402

from tradinglib.backtest import run_backtest  # noqa: E402
from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402

RESULTS = HERE / "results"
FEE_BPS = 1.0
SLIPPAGE_BPS = 0.5


def _get_or_train_bundle() -> dict[str, Any]:
    """Return the trained-model bundle, training on demand if missing."""
    bundle_path = RESULTS / "model.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    # First-time path (e.g. fresh Streamlit Cloud clone — joblib is gitignored).
    bars = load_daily(SYMBOL, start=START, end=END)
    bundle = train_model(bars["close"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    return bundle


def run_for_gui(
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    symbol: str = SYMBOL,
    fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    initial_capital: float = 100_000.0,
    size_mult: float = 1.0,
) -> dict[str, Any]:
    """Predict + backtest on a user-selected window.

    Inference-only: uses the model trained on 2010 → split. If the
    joblib is missing (fresh clone), it's trained once and cached. The
    GUI date window is intersected with the OOS slice (`split → END`)
    so we don't accidentally evaluate on training data.
    """
    if symbol != SYMBOL:
        raise ValueError(
            f"this model is locked to {SYMBOL} (the XGBoost model was trained "
            f"on it); retraining on {symbol!r} is out of scope for the GUI"
        )
    bundle = _get_or_train_bundle()
    model = bundle["model"]
    features = bundle["features"]
    split = bundle["split_index"]

    bars = load_daily(SYMBOL, start=START, end=END)
    prices = bars["close"]
    feats = build_features(prices)

    # Clip the requested window to the OOS slice to avoid training-data peeks
    start_ts = max(pd.Timestamp(start, tz="UTC"), split) if start is not None else split
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else feats.index.max()

    x = feats.loc[(feats.index >= start_ts) & (feats.index <= end_ts), features].dropna()
    if x.empty:
        raise ValueError(
            f"no OOS bars in [{start_ts.date()}, {end_ts.date()}] — the model's "
            f"OOS slice starts at {split.date()}"
        )
    pred = pd.Series(model.predict(x), index=x.index, name="predicted_return")
    signal = (pred > 0).astype(float) * size_mult

    user_prices = prices.loc[signal.index]
    user_opens = bars["open"].loc[signal.index]
    result = run_backtest(
        user_prices,
        signal,
        open_prices=user_opens,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
    )

    data = pd.DataFrame({"close": user_prices, "predicted_return": pred, "signal": signal})
    return {
        "data": data,
        "result": result,
        "symbol": SYMBOL,
        "params": {
            "start": str(start_ts.date()),
            "end": str(end_ts.date()),
            "oos_split": split.date().isoformat(),
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "size_mult": size_mult,
        },
    }


def main() -> None:
    bundle = _get_or_train_bundle()
    model = bundle["model"]
    features = bundle["features"]
    split = bundle["split_index"]

    bars = load_daily(SYMBOL, start=START, end=END)
    prices = bars["close"]
    feats = build_features(prices)

    x_oos = feats.loc[feats.index >= split, features].dropna()
    pred = pd.Series(model.predict(x_oos), index=x_oos.index)
    signal = (pred > 0).astype(float)
    oos_prices = prices.loc[signal.index]
    oos_opens = bars["open"].loc[signal.index]
    result = run_backtest(
        oos_prices,
        signal,
        open_prices=oos_opens,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
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
