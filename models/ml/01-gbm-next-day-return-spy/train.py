"""Train an XGBoost regressor to predict next-day SPY log return.

Trains on the first 80% of the historical window, evaluates on the held-out
last 20%. The trained model + feature list + split index get pickled to
``results/model.joblib`` so ``backtest.py`` can run a backtest on the OOS
window without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from tradinglib.features.technical import (
    log_return,
    price_to_sma_ratio,
    realized_volatility,
    rsi,
)
from tradinglib.loaders.equities.yfinance import load_daily

SYMBOL = "SPY"
START = "2010-01-01"
END = "2024-12-31"
TRAIN_FRAC = 0.8
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def build_features(prices: pd.Series) -> pd.DataFrame:
    """Construct the feature matrix for the model."""
    df = pd.DataFrame(index=prices.index)
    df["ret_1"] = log_return(prices, 1)
    df["ret_5"] = log_return(prices, 5)
    df["ret_20"] = log_return(prices, 20)
    df["vol_5"] = realized_volatility(df["ret_1"], 5)
    df["vol_20"] = realized_volatility(df["ret_1"], 20)
    df["rsi_14"] = rsi(prices, 14)
    df["px_vs_sma20"] = price_to_sma_ratio(prices, 20)
    df["px_vs_sma60"] = price_to_sma_ratio(prices, 60)
    return df


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    bars = load_daily(SYMBOL, start=START, end=END)
    prices = bars["close"]

    features = build_features(prices)
    target = log_return(prices, 1).shift(-1)  # next-day log return

    aligned = pd.concat([features, target.rename("y")], axis=1).dropna()
    n_train = int(len(aligned) * TRAIN_FRAC)
    train, test = aligned.iloc[:n_train], aligned.iloc[n_train:]

    feature_cols = [c for c in aligned.columns if c != "y"]
    x_train, y_train = train[feature_cols], train["y"]
    x_test, y_test = test[feature_cols], test["y"]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    summary = {
        "symbol": SYMBOL,
        "n_train": int(n_train),
        "n_test": int(len(aligned) - n_train),
        "split_date": aligned.index[n_train].date().isoformat(),
        "train_corr": float(np.corrcoef(pred_train, y_train)[0, 1]),
        "test_corr": float(np.corrcoef(pred_test, y_test)[0, 1]),
        "train_sign_accuracy": float((np.sign(pred_train) == np.sign(y_train)).mean()),
        "test_sign_accuracy": float((np.sign(pred_test) == np.sign(y_test)).mean()),
    }

    joblib.dump(
        {"model": model, "features": feature_cols, "split_index": aligned.index[n_train]},
        RESULTS / "model.joblib",
    )
    (RESULTS / "train_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
