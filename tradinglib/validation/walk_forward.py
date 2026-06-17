"""Walk-forward optimization harness.

Drives a model across rolling or anchored windows. On each window it selects the
best parameters by in-sample-window backtest Sharpe, applies them to the
out-of-sample window, and finally stitches all OOS slices into one continuous
run scored with the Deflated Sharpe deflated by the parameter-grid size.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from tradinglib.backtest.engine import BacktestResult, run_backtest
from tradinglib.backtest.resting_engine import run_resting_backtest
from tradinglib.validation.search import expand_grid, grid_search
from tradinglib.validation.splits import anchored_windows, rolling_windows

# make_signal(train_df, test_df, params) -> target-position series indexed like test_df
SignalFn = Callable[[pd.DataFrame, pd.DataFrame, dict], pd.Series]
# make_orders(train_df, test_df, params) -> resting-order frame indexed like test_df
# (columns entry/stop/target/entry_type/armed); the alternative scoring path (A2).
OrdersFn = Callable[[pd.DataFrame, pd.DataFrame, dict], pd.DataFrame]


@dataclass
class WalkForwardResult:
    oos_result: BacktestResult
    windows: pd.DataFrame
    param_stability: dict


def _in_sample_sharpe(
    train: pd.DataFrame,
    make_signal: SignalFn,
    params: dict,
    *,
    price_col: str,
    open_col: str,
    fee_bps: float,
    slippage_bps: float,
    periods_per_year: int,
) -> float:
    sig = make_signal(train, train, params)
    if not sig.index.equals(train.index):
        raise ValueError("make_signal must return a series indexed like the in-sample window")
    res = run_backtest(
        train[price_col],
        sig,
        open_prices=train[open_col],
        fill="next_open",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        periods_per_year=periods_per_year,
    )
    return float(res.metrics["sharpe"])


def _in_sample_resting_sharpe(
    train: pd.DataFrame,
    make_orders: OrdersFn,
    params: dict,
    *,
    stance: str,
    fee_bps: float,
    slippage_bps: float,
    periods_per_year: int,
) -> float:
    orders = make_orders(train, train, params)
    if not orders.index.equals(train.index):
        raise ValueError("make_orders must return a frame indexed like the in-sample window")
    res = run_resting_backtest(
        train,
        orders,
        stance=stance,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        periods_per_year=periods_per_year,
    )
    return float(res.metrics["sharpe"])


def walk_forward(
    data: pd.DataFrame,
    make_signal: SignalFn | None = None,
    *,
    param_grid: dict[str, list],
    mode: Literal["anchored", "rolling"] = "anchored",
    test_size: int,
    initial_train: int | None = None,
    train_size: int | None = None,
    step: int | None = None,
    embargo: int = 0,
    price_col: str = "close",
    open_col: str = "open",
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
    make_orders: OrdersFn | None = None,
    stance: str = "long",
) -> WalkForwardResult:
    # Exactly one scoring path: a {0, ±1} position series (make_signal, the
    # vectorized next-open engine) OR a resting stop/limit order filled intrabar
    # (make_orders, the resting engine — audit A2, cert==ticket).
    if (make_signal is None) == (make_orders is None):
        raise ValueError("pass exactly one of make_signal= or make_orders=")
    resting = make_orders is not None
    if price_col not in data or open_col not in data:
        raise ValueError(f"data must have '{price_col}' and '{open_col}' columns")
    if resting and not {"high", "low"}.issubset(data.columns):
        raise ValueError("the resting (make_orders) path requires 'high' and 'low' columns")

    if mode == "anchored":
        if initial_train is None:
            raise ValueError("mode='anchored' requires initial_train")
        windows = anchored_windows(data.index, initial_train, test_size, step, embargo=embargo)
    elif mode == "rolling":
        if train_size is None:
            raise ValueError("mode='rolling' requires train_size")
        windows = rolling_windows(data.index, train_size, test_size, step, embargo=embargo)
    else:
        raise ValueError(f"mode must be 'anchored' or 'rolling', got {mode!r}")
    if not windows:
        raise ValueError("no walk-forward windows — check sizes vs data length")

    n_trials = len(expand_grid(param_grid))
    if n_trials == 0:
        raise ValueError("param_grid produced an empty search space — check for empty value lists")
    oos_signals: list[pd.Series] = []
    oos_orders: list[pd.DataFrame] = []
    rows: list[dict] = []

    for train_idx, test_idx in windows:
        train = data.loc[train_idx]
        test = data.loc[test_idx]

        def _score(p: dict, _train: pd.DataFrame = train) -> float:
            if make_orders is not None:
                return _in_sample_resting_sharpe(
                    _train,
                    make_orders,
                    p,
                    stance=stance,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    periods_per_year=periods_per_year,
                )
            assert make_signal is not None  # guaranteed by the exactly-one check
            return _in_sample_sharpe(
                _train,
                make_signal,
                p,
                price_col=price_col,
                open_col=open_col,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                periods_per_year=periods_per_year,
            )

        search = grid_search(param_grid, _score)
        if make_orders is not None:
            orders = make_orders(train, test, search.best_params)
            if not orders.index.equals(test.index):
                raise ValueError("make_orders must return a frame indexed like test_df")
            oos_orders.append(orders)
        else:
            assert make_signal is not None  # guaranteed by the exactly-one check
            sig = make_signal(train, test, search.best_params)
            if not sig.index.equals(test.index):
                raise ValueError("make_signal must return a series indexed like test_df")
            oos_signals.append(sig)
        rows.append(
            {
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "test_start": test_idx[0],
                "test_end": test_idx[-1],
                "in_sample_sharpe": search.best_score,
                **{f"param_{k}": v for k, v in search.best_params.items()},
            }
        )

    if resting:
        oos_orders_df = pd.concat(oos_orders)
        if oos_orders_df.index.has_duplicates:
            raise ValueError("overlapping test windows (use step >= test_size)")
        oos_result = run_resting_backtest(
            data.loc[oos_orders_df.index],
            oos_orders_df,
            stance=stance,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
            n_trials=n_trials,
        )
    else:
        oos_signal = pd.concat(oos_signals)
        if oos_signal.index.has_duplicates:
            raise ValueError("overlapping test windows (use step >= test_size)")
        oos_result = run_backtest(
            data.loc[oos_signal.index, price_col],
            oos_signal,
            open_prices=data.loc[oos_signal.index, open_col],
            fill="next_open",
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
            n_trials=n_trials,
        )
    windows_df = pd.DataFrame(rows)
    return WalkForwardResult(
        oos_result=oos_result,
        windows=windows_df,
        param_stability=_param_stability(windows_df, param_grid),
    )


def _param_stability(windows_df: pd.DataFrame, param_grid: dict[str, list]) -> dict:
    """Fraction of window-to-window transitions where each param changed."""
    stability: dict = {}
    n_transitions = max(len(windows_df) - 1, 1)
    for k in param_grid:
        col = f"param_{k}"
        if col in windows_df:
            changes = int((windows_df[col] != windows_df[col].shift()).iloc[1:].sum())
            stability[k] = changes / n_transitions
    return stability
