"""Walk-forward re-fit of the XGBoost next-day-return model.

Re-fits the regressor on each anchored in-sample window (re-selecting depth /
n_estimators), predicts the OOS window, and reports the stitched OOS equity
deflated by the hyperparameter-grid size.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import END, START, SYMBOL, build_features  # noqa: E402

from tradinglib.features.technical import log_return  # noqa: E402
from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402
from tradinglib.validation import walk_forward  # noqa: E402

RESULTS = HERE / "results"
GRID = {"max_depth": [3, 4], "n_estimators": [200, 300]}
_WARMUP = 60  # bars of history needed to warm the rolling features


def make_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.Series:
    feats = build_features(train["close"])
    target = log_return(train["close"], 1).shift(-1)
    aligned = pd.concat([feats, target.rename("y")], axis=1).dropna()
    cols = [c for c in aligned.columns if c != "y"]

    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )
    model.fit(aligned[cols], aligned["y"], verbose=False)

    # Warm test features with the tail of train so rolling windows are valid.
    # Dedupe so in-sample scoring (test == train) doesn't double the index.
    warm = pd.concat([train["close"].tail(_WARMUP), test["close"]])
    warm = warm[~warm.index.duplicated(keep="last")]
    test_feats = build_features(warm).loc[test.index, cols].dropna()
    pred = pd.Series(model.predict(test_feats), index=test_feats.index)
    return (pred > 0).astype(float).reindex(test.index).fillna(0.0)


def main() -> None:
    bars = load_daily(SYMBOL, start=START, end=END)
    data = bars[["open", "close"]].dropna()

    wf = walk_forward(
        data, make_signal, param_grid=GRID, mode="anchored",
        initial_train=1008, test_size=126, step=126,  # ~4y train, ~6mo OOS
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_windows": len(wf.windows),
        "n_trials": int(wf.oos_result.config["n_trials"]),
        "walk_forward_oos": wf.oos_result.metrics,
        "param_stability": wf.param_stability,
    }
    (RESULTS / "walk_forward.json").write_text(json.dumps(summary, indent=2, default=str))
    wf.windows.to_csv(RESULTS / "walk_forward_windows.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    wf.oos_result.equity_curve.plot(ax=ax, label="XGBoost walk-forward (OOS)")
    ax.set_title(f"{SYMBOL} — XGBoost, walk-forward OOS")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS / "walk_forward_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
