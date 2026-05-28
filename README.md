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

The full sortable index lives in [MODELS.md](MODELS.md) and is
auto-generated from each model's `model.md` frontmatter.

## Roadmap

- **Alt-data model** — sentiment/search-interest signal applied to crypto.
  Loader needs careful design around historical data availability; see
  [`docs/data-sources.md`](docs/data-sources.md).
- **Microstructure model** — orderbook imbalance / microprice on a free
  crypto venue (Binance L2). Requires the event-driven backtest engine to
  land first.
- **Walk-forward CV** — moving from chronological 80/20 splits to rolling
  walk-forward as a default for the ML family.
- **Live data loaders** — Polygon and Alpaca for equities; Binance for
  crypto trades + L2.

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
