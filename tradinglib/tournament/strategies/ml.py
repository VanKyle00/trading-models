"""Machine-learned tournament strategies (fit on train inside make_signal).

The walk-forward signature already supports ML: ``make_signal(train, test,
params, stance)`` fits on ``train`` only and predicts over ``test``. Two
classic leaks are handled by construction — the next-bar-return target of the
last train row lives in the test window (that row is excluded from the fit),
and features are standardized with train-only statistics. The conformance
harness (tests/test_strategy_conformance.py) enforces both mechanically for
any future model. No RNG anywhere: closed-form solve, deterministic by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.features.technical import sma
from tradinglib.tournament.levels import Levels, direction, protective_stop, two_r_target
from tradinglib.tournament.strategies._core import StrategyDef, _full_history, register

_MIN_FIT_ROWS = 60


def _features(close: pd.Series) -> pd.DataFrame:
    r1 = close.pct_change()
    return pd.DataFrame(
        {
            "r1": r1,
            "r5": close.pct_change(5),
            "r20": close.pct_change(20),
            "vol20": r1.rolling(20).std(),
            "dist50": close / sma(close, 50) - 1.0,
            "dist200": close / sma(close, 200) - 1.0,
        }
    )


def _fit(train_close: pd.Series, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    feats = _features(train_close)
    # next-bar return WITHIN train: the last train row's target lives in the
    # test window, so shift(-1) makes it NaN and the dropna excludes it
    y = train_close.pct_change().shift(-1)
    frame = feats.assign(_y=y).dropna()
    if len(frame) < _MIN_FIT_ROWS:
        return None
    x = frame.drop(columns="_y").to_numpy(dtype=float)
    target = frame["_y"].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0  # a constant column contributes nothing, not a NaN
    xs = (x - mean) / std
    n_features = xs.shape[1]
    w = np.linalg.solve(xs.T @ xs + l2 * np.eye(n_features), xs.T @ target)
    return w, mean, std


def _predict(feats: pd.DataFrame, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> pd.Series:
    w, mean, std = fit
    x = feats.to_numpy(dtype=float)
    pred = ((x - mean) / std) @ w
    out = pd.Series(pred, index=feats.index)
    return out.where(feats.notna().all(axis=1))  # warmup rows predict nothing


def _ridge_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict, stance: str) -> pd.Series:
    full = _full_history(train, test)
    fit = _fit(train["close"], params["l2"])
    if fit is None:
        return pd.Series(0.0, index=test.index)
    pred = _predict(_features(full["close"]), fit)
    pos = (pred > 0.0).astype(float) if direction(stance) > 0 else -(pred < 0.0).astype(float)
    return pos.loc[test.index]


def _ridge_levels(bars: pd.DataFrame, params: dict, stance: str) -> Levels | None:
    fit = _fit(bars["close"], params["l2"])
    if fit is None:
        return None
    pred = _predict(_features(bars["close"]), fit)
    last = pred.iloc[-1]
    if pd.isna(last) or not direction(stance) * float(last) > 0.0:
        return None  # the model does not predict a favorable return for this stance
    entry = float(bars["close"].iloc[-1])
    try:
        stop = protective_stop(bars, entry, stance)
    except ValueError:
        return None  # degenerate ATR -> no stop distance, no actionable entry
    return Levels(
        entry=entry,
        entry_type="market",
        stop=stop,
        target=two_r_target(entry, stop),
        condition=(
            f"ridge(l2={params['l2']:g}) predicts a favorable next-bar return for "
            f"this stance; market entry with a 2xATR protective stop"
        ),
    )


register(
    StrategyDef(
        key="ridge_momentum",
        name="Ridge momentum",
        style="ml",
        description=(
            "Closed-form ridge regression on lagged returns, volatility, and SMA "
            "distances, fit per walk-forward window on train bars only; long while "
            "the predicted next-bar return is positive (short stance mirrors)."
        ),
        param_grid={"l2": [1.0, 10.0]},
        make_signal=_ridge_signal,
        levels=_ridge_levels,
    )
)
