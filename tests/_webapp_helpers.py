# tests/_webapp_helpers.py
import numpy as np
import pandas as pd

from tradinglib.backtest import run_backtest
from tradinglib.service.run import BacktestRun


def synthetic_run(n: int = 300) -> BacktestRun:
    """A deterministic BacktestRun for chart/serialization unit tests."""
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n)), index=idx)
    signal = (prices > prices.rolling(20).mean()).astype(float).fillna(0.0)
    result = run_backtest(prices, signal, fill="decision_close", fee_bps=1.0, slippage_bps=0.5)
    data = pd.DataFrame({"close": prices, "signal": signal})
    return BacktestRun(
        model_id="models/classical/01-sma-crossover-spy",
        symbol="SYN",
        params={"start": str(idx[0].date()), "end": str(idx[-1].date())},
        result=result,
        data=data,
    )
