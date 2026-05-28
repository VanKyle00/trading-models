# 01 — XGBoost Next-Day Return on SPY

**Hypothesis.** A boosted tree model can learn enough non-linear structure
in standard technical features (recent returns, volatility, RSI, deviation
from moving averages) to produce a directional next-day-return signal that
beats a passive long position in regimes where the OOS sample sees
sufficient drawdowns.

**Data.** SPY daily adjusted bars from yfinance, `2010-01-01` to
`2024-12-31`.

**Features** (eight scalar features per bar):

| Feature | Description |
| --- | --- |
| `ret_1`, `ret_5`, `ret_20` | Log return over 1 / 5 / 20 bars |
| `vol_5`, `vol_20` | Rolling std of `ret_1` over 5 / 20 bars |
| `rsi_14` | 14-bar Relative Strength Index |
| `px_vs_sma20`, `px_vs_sma60` | Price deviation from 20 / 60-bar SMA |

**Target.** Next-bar log return (regression).

**Model.** `xgboost.XGBRegressor`, 300 trees, depth 4, learning rate 0.03,
80/20 column and row subsampling, `random_state=42`. Hyperparameters are
deliberately conservative to keep the model honest — no
tuning-on-the-test-set wizardry.

**Train / test split.** Chronological 80% / 20%. No shuffling — that would
leak information across time and produce wildly optimistic metrics.

**Trading rule.** Long when predicted next-day log return > 0, flat
otherwise.

**Costs.** 1 bp commission + 0.5 bp slippage per unit of turnover.

## Reproduce

```bash
uv run python models/ml/01-gbm-next-day-return-spy/train.py
uv run python models/ml/01-gbm-next-day-return-spy/backtest.py
```

`train.py` fits the model on the 80% training window, saves it to
[`results/model.joblib`](results/model.joblib), and writes training stats
to [`results/train_metrics.json`](results/train_metrics.json).
`backtest.py` loads the model, runs the standard backtest engine on the
OOS window, and writes [`results/metrics.json`](results/metrics.json) and
[`results/equity_curve.png`](results/equity_curve.png).

## Results

See `results/metrics.json` for OOS Sharpe, drawdown, hit rate, and turnover.
The accompanying plot compares the strategy against a buy-and-hold
benchmark over the same OOS window.

## Notes / takeaways

_To be filled in after the first run — discuss whether the OOS signal
actually has predictive power, what regimes it handles well/poorly, and
ideas for the next iteration._
