"""Parameter-sensitivity and regime-breakdown diagnostics.

``parameter_sensitivity`` evaluates every grid config on a fixed out-of-sample
window so a chosen config can be read against its neighbours (plateau = robust,
lonely spike = overfit). ``metrics_by_regime`` reports ``compute_metrics`` per
sub-period (calendar year or a supplied label series).
"""
from __future__ import annotations

import pandas as pd

from tradinglib.backtest.engine import run_backtest
from tradinglib.backtest.metrics import compute_metrics
from tradinglib.validation.search import expand_grid
from tradinglib.validation.walk_forward import SignalFn


def parameter_sensitivity(
    data: pd.DataFrame,
    make_signal: SignalFn,
    param_grid: dict[str, list],
    *,
    train_index: pd.Index,
    test_index: pd.Index,
    price_col: str = "close",
    open_col: str = "open",
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Evaluate every grid config on a fixed OOS window -> tidy metrics frame."""
    train = data.loc[train_index]
    test = data.loc[test_index]
    rows: list[dict] = []
    for params in expand_grid(param_grid):
        sig = make_signal(train, test, params)
        res = run_backtest(
            test[price_col], sig, open_prices=test[open_col], fill="next_open",
            fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year,
        )
        rows.append({
            **params,
            "sharpe": res.metrics["sharpe"],
            "annualized_return": res.metrics["annualized_return"],
            "max_drawdown": res.metrics["max_drawdown"],
        })
    return pd.DataFrame(rows)


def vol_regime(prices: pd.Series, window: int = 20, n_bins: int = 3) -> pd.Series:
    """Label each bar by trailing-volatility quantile bucket (vol_q0 = calmest)."""
    vol = prices.pct_change().rolling(window).std()
    labels = pd.qcut(vol, n_bins, labels=[f"vol_q{i}" for i in range(n_bins)])
    return pd.Series(labels, index=prices.index, name="regime")


def metrics_by_regime(
    returns: pd.Series,
    equity_curve: pd.Series,
    *,
    by: str | pd.Series = "year",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """compute_metrics per sub-period (calendar 'year' or a label series).

    Bars whose label is NaN (e.g. the warm-up bars from ``vol_regime``) are
    excluded by the groupby, so per-regime ``n_bars`` may sum to fewer than
    ``len(returns)``.
    """
    if isinstance(by, pd.Series):
        labels = by.reindex(returns.index)
    elif by == "year":
        labels = pd.Series(returns.index.year, index=returns.index, name="regime")
    else:
        raise ValueError("by must be 'year' or a label Series")

    rows: list[dict] = []
    for label, grp in returns.groupby(labels):
        eq = equity_curve.loc[grp.index]
        rows.append({"regime": label, **compute_metrics(grp, eq, periods_per_year=periods_per_year)})
    return pd.DataFrame(rows).set_index("regime")
