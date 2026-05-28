# Data

This directory holds **data ingestion code** (under `ingestion/`) and the
**downloaded data itself** (under `raw/` and `processed/`, both gitignored).

## Data catalog

| Source | Asset class | Type | Loader | Free? |
| --- | --- | --- | --- | --- |
| yfinance | equities | OHLCV bars | [`ingestion/equities/yfinance.py`](ingestion/equities/yfinance.py) | yes |

_(more sources to come)_

## Layout

```
data/
├── ingestion/         # Code: one module per (asset_class, source) pair
├── raw/               # gitignored — exactly what the source returned
└── processed/         # gitignored — canonicalized parquet, one file per (source, symbol, date)
```

## Conventions

- Every loader reads from its source and writes parquet to
  `data/processed/<source>/<symbol>/<YYYY-MM-DD>.parquet`.
- DuckDB queries parquet files in place — there is no separate database.
- Secrets live in `.env` (gitignored). See `.env.example` at the repo root for
  the keys each source needs.
- Every loader module ships a `README.md` documenting the source's schema,
  rate limits, and license.
