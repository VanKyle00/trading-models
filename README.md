# Trading Models

A growing portfolio of trading models spanning classical quant, machine
learning, market microstructure, and alternative data — across US equities
and crypto.

Each model lives in a self-contained directory under `models/<family>/`
with a reproducible notebook, a `backtest.py` entry point, and standardized
metrics. The shared library in `tradinglib/` provides a unified backtest
engine so every model is measured the same way and the results in the
table below are directly comparable.

## Current models

| Model | Family | Window | Assets | OOS Sharpe | Max DD | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [SMA Crossover on SPY](models/classical/01-sma-crossover-spy/) | classical | swing | equities | 0.73 | -0.34 | working |
| [XGBoost Next-Day Return on SPY](models/ml/01-gbm-next-day-return-spy/) | ml | swing | equities | 0.84 | -0.14 | working |
| [Google Trends Contrarian on BTC](models/alt-data/01-google-trends-btc/) | alt-data | swing | crypto | -0.30 | -0.80 | negative-result |

The third row is intentional: a hypothesis (search interest as a contrarian
sentiment signal) was tested honestly and the data rejected it. See the
model's README for the full writeup and what the inverse direction looks
like. Negative results are first-class citizens here — the alternative is a
portfolio of overfit "winners".

The full sortable index lives in [MODELS.md](MODELS.md) and is
auto-generated from each model's `model.md` frontmatter.

## Roadmap

- **Microstructure seed model** — orderbook imbalance / microprice on
  Binance L2. The two prerequisites (Binance loader + event-driven
  backtest engine) are now in; the remaining piece is a tick / L2
  ingestion path (WebSocket capture → parquet).
- **Walk-forward CV** — moving from chronological 80/20 splits to rolling
  walk-forward as the default for the ML family.
- **Vendor-grade equities loaders** — Polygon or Alpaca to replace
  yfinance for production-quality bars.
- **More alt-data sources** — the Trends contrarian model was a negative
  result; NewsAPI + a proper headline-sentiment model is the next
  experiment worth running.

## Repository tour

| Directory | What lives there |
| --- | --- |
| `tradinglib/` | Shared package — data, features, backtest engine, metrics, viz |
| `tradinglib/loaders/` | Data loaders, one subpackage per asset class |
| `data/ingestion/` | Documentation of each data source |
| `models/classical/` | Mean reversion, momentum, pairs trading, statistical arbitrage |
| `models/ml/` | Gradient boosting, LSTMs, transformers |
| `models/microstructure/` | Orderbook imbalance, microprice, queue dynamics (empty for now) |
| `models/alt-data/` | Sentiment, news, alternative signals (empty for now) |
| `notebooks/eda/` | Exploratory analyses not tied to a single model |
| `docs/` | Glossary, data sources, backtest methodology, latency notes |
| `scripts/` | Maintenance scripts (e.g., regenerating `MODELS.md`) |
| `tests/` | Unit tests for `tradinglib` |

## Quick start

```bash
git clone https://github.com/<you>/trading-models.git
cd trading-models
uv sync --all-extras
cp .env.example .env   # fill in any API keys you need (none required for the seed models)
```

Run the tests:

```bash
uv run pytest
```

Reproduce a model's backtest:

```bash
uv run python models/classical/01-sma-crossover-spy/backtest.py
```

Train + backtest the ML model:

```bash
uv run python models/ml/01-gbm-next-day-return-spy/train.py
uv run python models/ml/01-gbm-next-day-return-spy/backtest.py
```

Add a new model? Drop it under `models/<family>/NN-slug/` with a
`model.md` frontmatter block, then regenerate the index:

```bash
uv run python scripts/regenerate_models_index.py
```

## Methodology

Every model is evaluated with the same backtest engine
(`tradinglib.backtest`) and reports the same metrics: annualized return,
Sharpe ratio, Sortino ratio, maximum drawdown, hit rate, and turnover.
Assumptions about slippage, transaction costs, look-ahead bias prevention,
and the train/test split discipline are documented in
[`docs/methodology.md`](docs/methodology.md).

New to systematic trading? Start with the
[**glossary**](docs/glossary.md) — terms are defined plainly with the
context needed to follow the rest of the repo. Wondering when you'd need to
leave Python for C++? See [`docs/latency-notes.md`](docs/latency-notes.md).

## Status

Foundation + two seed models live. The structure is ready to absorb
additional models in any of the four families — see the roadmap above for
what's next.
