# Planner after-hours chain fallback (CBOE) — implementation plan

Design: `docs/specs/2026-06-11-planner-cboe-afterhours-fallback-design.md`

TDD throughout — every step is test-first against mocked HTTP (repo
convention: no live network in tests).

## Step 1: CBOE loader — `tradinglib/loaders/options/cboe_chain.py`

Tests (`tests/test_cboe_chain.py`), then implement:

1. `fetch_cboe_chain` maps a faked payload to the canonical `CHAIN_COLUMNS`
   frame: OCC tail parsed (expiration/right/strike), bid/ask/iv/OI/volume
   floats, spot from `current_price`, `date` = today ET.
2. Window: expirations beyond `max_days` and already-expired series excluded.
3. Symbology: `BRK-B` requested as `BRK.B`; `ticker` column keeps `BRK-B`.
4. Malformed `option` symbols are skipped, others survive.
5. HTTP/fetch errors raise (no silent empty frame).

→ verify: `uv run pytest tests/test_cboe_chain.py` red, implement, green.

## Step 2: planner fallback — `tradinglib/assistant/planner.py`

Tests (extend `tests/test_assistant_planner.py`), then implement
`_chain_with_fallback` + warning:

1. Degenerate yf chain (all zero-bid) + healthy CBOE chain → ticket builds,
   stale-quotes warning present, structures carry `calculator_url`.
2. Healthy yf chain → CBOE never called, no stale warning.
3. Degenerate yf chain + CBOE raises → original
   `no option structure passed the liquidity gate` error surfaces.
4. `fetch_chain` raises + healthy CBOE → ticket builds with warning
   (yfinance-429 rescue).
5. Empty yf chain → fallback used.

→ verify: `uv run pytest tests/test_assistant_planner.py` red → green.

## Step 3: full gate + live verification

- Six-step gate: `ruff check`, `ruff format --check`, `mypy tradinglib`,
  `pytest`, streamlit import check, MODELS.md freshness.
- Off-hours end-to-end: rerun `data/tmp/repro_planner_link.py` (real LLM +
  real network at ~4 AM ET) — expect a ticket card payload with an
  optionstrat.com `calculator_url` and the stale-quotes warning.

## Step 4: PR

Single PR to `main` (no stack). README "Current models" untouched (no new
model). Commits: `feat(loaders): CBOE delayed-quotes chain loader` +
`feat(planner): fall back to CBOE chain when yfinance quotes are zeroed`.
