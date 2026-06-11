# Planner after-hours chain fallback (CBOE delayed quotes) — design

**Date:** 2026-06-11
**Branch:** `feat/planner-cboe-afterhours-fallback` (based on `main` directly — no stack).

## Problem

Outside regular trading hours the planner cannot build a ticket at all: every
`build_options_ticket` call fails with `no option structure passed the
liquidity gate`, so the chat recommends nothing and no OptionStrat link is
produced. Reproduced 2026-06-11 ~03:40 ET on AAPL:

- yfinance chain: 1,187/1,210 rows with `bid == 0`, open interest zeroed
  almost everywhere — **3/1,210** rows pass the liquidity gate
  (`bid > 0`, spread ≤ 10 % of mid, OI ≥ 100, IV > 0).
- Root cause is the data source, not the gate: US equity options do not trade
  overnight, and Yahoo *discards* the closing bid/ask (and OI) when the market
  closes. Other feeds keep the close snapshot available.

CBOE's public delayed-quotes CDN retains the last session's closing quotes
around the clock. Probed at the same time, same ticker:
`https://cdn.cboe.com/api/global/delayed_quotes/options/AAPL.json` →
3,275/3,712 rows with `bid > 0`, real OI, **1,060/3,712** pass the full gate.
One request returns the whole board (vs. yfinance's request-per-expiration —
also sidesteps the 429 problem).

## Fix: CBOE loader + degenerate-chain fallback in the planner

### New loader `tradinglib/loaders/options/cboe_chain.py`

`fetch_cboe_chain(ticker, *, max_days=FETCH_MAX_DAYS) -> pd.DataFrame` in the
canonical `CHAIN_COLUMNS` schema (`[date, ticker, expiration, strike, right,
bid, ask, iv, spot, open_interest, volume]`), in-memory only, never persisted.

Payload mapping (probed live):

- Request symbol: Yahoo-style dashes become dots — `BRK-B` → `BRK.B`
  (`BRK-B`/`BRKB` return HTTP 403). Returned `ticker` column keeps the
  caller's (Yahoo-style) spelling.
- Per-record `option` field is OCC-style with the dot stripped
  (`BRKB260612C00270000`). Expiration/right/strike parse from the fixed
  15-char tail: `(\d{6})([CP])(\d{8})$`, strike = int/1000. Malformed symbols
  are skipped, not fatal.
- `bid`/`ask`/`iv`/`open_interest`/`volume` map 1:1; spot =
  `data.current_price`; `date` = today's ET date (same convention as
  `yf_chain.fetch_chain`, keeps DTE math correct overnight).
- CBOE lists the full board out to LEAPS and includes just-expired series →
  keep only `today <= expiration <= today + max_days`.
- Fetch/HTTP errors raise (callers decide; the planner falls back to the
  yfinance behavior).

### Fallback hook in `tradinglib/assistant/planner.py`

`hypothesis_ticket` currently calls `fetch_chain(ticker)` directly. New
private helper:

```python
chain, stale_quotes = _chain_with_fallback(ticker)
```

- Healthy yfinance chain (≥ 20 % of rows with `bid > 0` —
  `_MIN_LIVE_BID_FRAC`) → use it, no CBOE request at all.
- Degenerate (below threshold, empty, or `fetch_chain` raised — the 429 case)
  → try CBOE. CBOE healthy → use it, `stale_quotes=True`. CBOE fails/empty →
  keep the original chain/error (behavior unchanged from today).
- When `stale_quotes`, append to the ticket warnings:
  `"live quotes unavailable (market closed?) — priced off CBOE delayed quotes
  from the last session; re-verify at the open"`. The model already relays
  warnings verbatim; the card renders them as ⚠ rows.

`propose_levels` never touches the chain — unchanged. The nightly scanner
(`scanner/pipeline.py`, the other `fetch_chain` caller) is deliberately NOT
wired in this change: it has a stock-plan fallback and its tickets feed the
forward ledger, so switching its quote source deserves its own look at ledger
comparability. Follow-up candidate.

## Non-goals

- No persistence of CBOE snapshots, no change to the snapshot collector.
- No change to the liquidity gate thresholds.
- No prompt changes: warnings ride the existing plumbing.

## Risks

- Unofficial public feed (same class as yfinance). Mitigation: it is the
  fallback, not the primary; failure degrades to today's behavior.
- Delayed quotes are the close, not the open — tickets built overnight can be
  stale by the open. Mitigated by the explicit warning + the existing
  `INDICATIVE_WARNING` ("re-check at the open, use limit orders") and
  `quotes_asof` chip.
