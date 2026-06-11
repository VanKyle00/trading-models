# Planner sizing settings + NaN-bar loader fix — design

**Date:** 2026-06-11
**Branch:** stacked on `feat/planner-guided-levels` (PR #66); retarget to `main` after #66 merges (squash-merge stack gotcha: retarget before deleting the base branch).

## Problem

Two complaints from using the /planner workbench:

1. **The chat asks for account size and risk% every conversation.** The server is
   stateless and the only place those numbers exist is the dialogue, so every new
   session re-collects them.
2. **The chat asks the user for entry/stop prices instead of proposing them.**
   PR #66 already added `propose_trade_levels` (ATR- and structure-grounded
   scenarios + chart card), but the tool currently fails with
   `degenerate ATR (nan); cannot place levels`, and the model falls back to
   interrogating the user. Reproduced root cause: yfinance returns the most
   recent session (2026-06-10 at time of writing) with volume populated but
   **NaN open/high/low/close** — for every ticker (verified on RIVN and SPY).
   The NaN row makes `rolling(14).mean()` of the true range NaN and makes
   `spot` (last close) NaN.

## Fix 1: drop NaN-price rows in `load_daily`

`tradinglib/loaders/equities/yfinance.py::load_daily` — after reading the frame
(from the parquet cache or a fresh download) and before the start/end filters,
drop rows where any of open/high/low/close is NaN:

```python
df = df.dropna(subset=["open", "high", "low", "close"])
```

- Filtering on the **read path** (not in `_canonicalize`) is deliberate: it also
  heals parquet caches that were already poisoned with the NaN row, and it keeps
  protecting consumers if yfinance keeps serving the bad row on refresh.
- Protects every consumer at once: planner, scanner, tournament, ledger.
- Docstring gains one line noting yfinance occasionally emits a partial trailing
  row (volume, no prices).
- Effect on the planner: `spot` becomes the last *complete* close, ATR(14) is
  finite, `propose_trade_levels` succeeds, and the dialogue proposes TA-based
  levels as designed in #66.

## Fix 2: sizing settings strip on /planner

### UI (`webapp/templates/planner.html`)

A slim settings bar between the strapline and the console, matching the page
chrome (Martian Mono labels, `--field` inputs):

```
ACCOUNT $ [100000]   RISK/TRADE % [1.0]   — sizes every ticket; the chat won't ask
```

- Two `<input type="number">` fields wrapped in a `#planner-settings` container:
  `account_size` (dollars, default 100000) and `risk_pct` (percent, default 1.0;
  converted to a fraction client-side before sending).
- Values persist to `localStorage` (`tm-planner-account`, `tm-planner-risk`) on
  change, restored on load — same pattern as the theme toggle. No save button;
  the current field values are read at send time.
- /planner only. The index console has no strip and its behavior is unchanged.

### Transport (`webapp/templates/_console.html`)

The submit handler reads the optional `#planner-settings` container. When
present, the POST body gains:

```json
"settings": {"account_size": 100000, "risk_per_trade_pct": 0.01}
```

Invalid/empty fields → the key is omitted (server treats it as absent).

### Server (`webapp/main.py`)

New `_planner_settings(raw) -> str | None` validator, mirroring `_chat_context`:
returns `None` for anything malformed (missing keys, non-numbers,
`account_size <= 0`, `risk_per_trade_pct` outside `(0, 0.2]`) so a bad payload
degrades to the no-settings flow rather than 400ing the chat. On success it
renders one line, e.g.:

```
Planner sizing (set on the page): account size $100,000; risk per trade 1% (0.01).
```

The chat route passes it to `run_chat(..., settings=...)`.

### Agent (`tradinglib/assistant/agent.py`)

`run_chat` gains an optional `settings: str | None` parameter. When present it
is appended to the opening message (after the `context` fold, same mechanism):

```
<settings line>
Use these for sizing; do not ask the user for account size or risk.
```

### Prompt (`tradinglib/assistant/provider.py`)

Planner step 3 of `SYSTEM_PROMPT` is amended: when sizing settings are present
in the conversation, use them for `build_options_ticket` and never ask about
account size or risk — the single bundled confirmation becomes scenario-only
("balanced / tight / structure — or 'go'", keeping user-tweaked numbers). When
no settings are present (index console), the current bundled
scenario+sizing question with $100k/1% defaults stands.

## Decided trade-offs

- **Stateless server preserved**: settings ride each request like `history`
  does; the browser is the source of truth. No settings endpoint, no session.
- **Prompt-layer injection, not tool-layer**: `dispatch` stays pure; the
  schema's `account_size`/`risk_per_trade_pct` stay required on
  `build_options_ticket`; the model fills them from the opening message.
- **Confirm flow**: one scenario-only confirmation before pulling live chain
  quotes (user-chosen over fully-automatic).

## Tests

1. **Loader**: seed a synthetic cached parquet whose trailing row has NaN
   prices (volume populated) → `load_daily` returns only complete bars; ATR
   over the result is finite. No-network test (cache path, `refresh=False`).
2. **Agent**: `run_chat(..., settings="...")` folds the settings line into the
   opening user message (StubProvider captures the conversation).
3. **Webapp**: `/api/v1/chat` with a valid `settings` object reaches the
   provider with the settings line; malformed `settings` (negative account,
   risk > 0.2, junk types) is ignored, not a 400.
4. **Template**: /planner page renders the `#planner-settings` strip with both
   inputs; index page does not.

## Out of scope

- Per-ticker or per-strategy sizing, server-side persistence, auth/profiles.
- Scanner/tournament sizing paths (they have their own config).
- Changes to ticket math — `build_options_ticket` semantics are untouched.
