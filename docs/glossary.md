# Glossary

Terms used throughout the repo, in roughly the order they show up. Aimed at
someone new to systematic trading.

## Market structure

- **Bar / candle** — A summary of price action over a fixed time interval
  (e.g., one day, one minute). Each bar has an *open*, *high*, *low*,
  *close*, and *volume* — collectively "OHLCV".
- **Tick** — One individual trade or quote update. Tick data is far higher
  resolution (and far larger) than bar data.
- **Limit order book (LOB)** — The current list of resting buy and sell
  limit orders at each price level. The "shape" of the book — depth on each
  side, distance between bid and ask — carries predictive information that
  microstructure models try to exploit.
- **Bid / ask spread** — The gap between the best buy price (bid) and the
  best sell price (ask). A market-buy order pays the ask; a market-sell
  hits the bid. Spread is an unavoidable cost of trading.
- **Slippage** — The difference between the price you expected to trade at
  and the price you actually got. Driven by bid/ask spread, market impact,
  and how fast prices move while your order is in flight.

## Strategy categories

- **Classical quant** — Strategies grounded in statistics and economic
  reasoning: mean reversion, momentum, pairs trading, statistical arbitrage.
  Tend to be explainable and have decades of academic backing.
- **ML / machine learning** — Use supervised models (gradient boosting,
  neural nets) to predict future price moves or returns from features.
  Higher capacity, less interpretable, more prone to overfit.
- **Microstructure** — Operate on the limit order book and tick data,
  predicting short-term price moves from book imbalance, queue dynamics,
  and similar microstructure features. Time horizon: seconds to minutes.
- **Alternative data** — Use non-price inputs (news sentiment, social media,
  satellite, web scrapes, etc.) as predictive signals. Often combined with
  traditional features.

## Trading windows

- **HFT (high-frequency trading)** — Sub-second holding periods. Needs
  colocation, custom hardware, often C++ / FPGA. Out of reach for a
  portfolio repo without exchange access.
- **Intraday** — Holding periods of minutes to hours within a single
  trading session.
- **Swing** — Days to a few weeks.
- **Long-term** — Weeks to months.

## Performance metrics

- **Annualized return** — Compound growth rate expressed per year. Lets
  you compare strategies that ran over different windows.
- **Sharpe ratio** — Mean return divided by return volatility, scaled to a
  year. The standard risk-adjusted return measure. >1 is decent for a
  long-only strategy; >2 is excellent; institutional alpha shops aim for >3.
- **Sortino ratio** — Like Sharpe, but only penalizes downside volatility.
  Distinguishes "swings up" (good) from "swings down" (bad).
- **Maximum drawdown** — The largest peak-to-trough decline in equity over
  the backtest. -10% is mild; -30% is serious; -50%+ is catastrophic.
- **Hit rate** — Fraction of bars (or trades) that were profitable. A high
  Sharpe with a low hit rate means the strategy has few big winners and
  many small losers — common for trend-following.
- **Turnover** — How frequently the strategy reshuffles positions. High
  turnover means costs eat more of the return.

## Backtesting hazards

- **Look-ahead bias** — Accidentally using information at bar `t` that
  wouldn't have been known until `t+1`. The most common backtest
  bug. This repo's engine lags every signal by one bar to prevent it.
- **Survivorship bias** — Backtesting only on symbols that survived to today.
  If the data set excludes delisted companies, returns are biased upward.
- **Overfitting** — Tuning hyperparameters until the backtest looks
  beautiful. The model has memorized noise; live performance collapses.
  Mitigated by walk-forward validation, parameter stability checks, and
  not torturing the data until it confesses.
- **Data snooping** — Trying many strategies until one looks good by
  chance. The repo's tests/metrics framework is designed to make every
  comparison apples-to-apples so this is harder to hide.

## ML terminology

- **Features (covariates / Xs)** — Inputs to a model.
- **Target (label / y)** — What the model is trying to predict (e.g.,
  next-bar return).
- **Train/test split** — Carving the data into one slice to fit the model
  on and a separate slice to evaluate on. **Always chronological** for
  time series — shuffling leaks the future into the past.
- **Walk-forward validation** — Repeatedly train on a window, predict the
  next window, slide forward. The realistic analog of cross-validation for
  time-series strategies.
- **Out-of-sample (OOS)** — Data the model never saw during training.
  The only honest measure of a model's predictive power.

## Risk / portfolio

- **Position** — How much of an asset you hold. In this repo, expressed as
  a multiple of equity: 1.0 = fully long, -1.0 = fully short, 0.0 = flat.
- **Leverage** — Position size > 1.0 (long) or < -1.0 (short). Magnifies
  returns and drawdowns.
- **Drawdown** — Equity has fallen from its prior peak; you're "in
  drawdown" until equity makes a new high.
- **Risk-free rate** — The yield on a riskless asset (typically short-term
  Treasuries). Sharpe is traditionally computed against this; this repo
  uses 0 for simplicity.
