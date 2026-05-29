# Configurable Backtesting GUI ("GUI v2") — Design

**Date:** 2026-05-28
**Status:** Approved (pending spec review)

## Goal

Extend the Streamlit app (`app/streamlit_app.py`) from a fixed-ticker,
date-window backtest viewer into a configurable backtesting workbench:
switch tickers, stress historical data with preset financial scenarios,
adjust execution/strategy parameters, and inspect results through richer
visualizations.

## Scope (confirmed with user)

1. **Ticker switching** — per-model whitelist.
2. **Scenario engine** — flash crash, volatility scaling, trend/drift,
   and return-shuffle (Monte Carlo Permutation Test). Gap/illiquidity is
   folded into the cost controls rather than the scenario engine. Extend
   the existing date-window presets with 2008 GFC / 2010 flash crash /
   2018 Q4.
3. **Visualizations** — rolling Sharpe + returns histogram, monthly
   returns heatmap, trade table & stats, exposure/position timeline.
4. **Backtest controls** — execution costs (fee/slippage/initial
   capital), per-model strategy params, position sizing.
5. **Comparison** — optional "compare against baseline" toggle.

### Out of scope (this pass)

- Retraining the ML model on arbitrary tickers (it stays locked to SPY).
- Price-transform scenarios on the trade-level microstructure model
  (it gets cost/sizing controls only).
- Bootstrap resampling and random trade-skipping (the "full battery"
  option was not selected).

## Research basis (scenario engine)

Three philosophies of "modifying historical data" inform the design:

- **Deterministic shocks** — "what if a specific bad thing happened?"
  Intuitive and visual: crash, vol-scale, drift, cost widening.
- **Statistical / Monte Carlo perturbation** — "is my edge real or
  curve-fit to noise?" The **return shuffle (MCPT)** is the most
  diagnostic single test: shuffle log-returns to destroy serial
  structure; if the strategy still profits, its edge was noise-fitting.
- **Historical event replay** — "how did it behave in real crises?"
  Already served by the date-window presets; extended here.

Sources:
- [Backtesting Algorithmic Futures Strategies — QuantStrategy.io](https://quantstrategy.io/blog/backtesting-algorithmic-futures-strategies-avoiding-curve/)
- [Monte Carlo Permutation Test — BuildAlpha](https://www.buildalpha.com/monte-carlo-permutation/)
- [Monte Carlo Simulations in Trading — QuantProof](https://quantproof.io/blog/monte-carlo-simulations-trading-strategy-validation)

---

## Architecture

### 1. Foundation: model metadata + standardized `run_for_gui`

Today each model hardcodes `SYMBOL`/params as module constants, and
`run_for_gui(start, end, **params)` differs per model. Every other
workstream depends on standardizing this first.

**Per-model metadata moves into `model.md` frontmatter.** It is already
parsed generically by `tradinglib/models_index.py::parse_frontmatter`,
so no parser changes are required:

```yaml
tickers: [SPY, QQQ, AAPL, TLT]   # explicit list,
                                 # or `tickers: any` (free-text yfinance),
                                 # or a single locked symbol e.g. [BTCUSDT]
scenario_capable: true           # false for the trade-level microstructure model
params:
  - {name: fast, label: Fast SMA, type: int, default: 50,  min: 5,  max: 150}
  - {name: slow, label: Slow SMA, type: int, default: 200, min: 50, max: 400}
```

The sidebar renders ticker and parameter widgets **dynamically** from
this metadata — model-specific knowledge stays in the model, not baked
into the GUI.

**Standardized signature** across all four models:

```python
def run_for_gui(
    start, end, *,
    symbol=DEFAULT_SYMBOL,
    scenario=None,            # tradinglib.scenarios.Scenario | None
    fee_bps=DEFAULT_FEE,
    slippage_bps=DEFAULT_SLIPPAGE,
    initial_capital=DEFAULT_CAPITAL,
    size_mult=1.0,
    **strategy_params,
) -> dict: ...
```

Each model's body follows the same shape:
`load(symbol) → scenario.apply(prices) → build_signal(**strategy_params)
→ run_backtest(costs, sizing)`. The adapter (`app/adapters.py`) already
forwards `**params`, so it just passes the assembled config dict; no
adapter change beyond constructing that dict.

The returned dict keeps its existing keys (`data`, `result`, `symbol`,
`params`) so the existing UI panels continue to work. `params` is
extended to echo the new config so the Data-details panel reports it.

**Family-specific handling:**

| Family | `tickers` | `scenario_capable` | Notes |
|---|---|---|---|
| classical | `any` | true | Free-text yfinance ticker; full scenarios. |
| ml | `[SPY]` (locked) | true (price leg) | Trained on SPY; switching requires retrain → out of scope. Caption explains the lock. |
| microstructure | `[BTCUSDT]` (locked) | **false** | Binance trade-level data; price transforms don't map onto OFI. Cost/sizing controls still apply. |
| alt-data | `[BTC-USD]` (locked) | true (price leg) | Google-Trends-for-BTC companion series is fixed; scenarios apply to the price leg. |

### 2. Scenario engine — `tradinglib/scenarios/`

A new, pure, independently testable module (primary TDD target).
Transforms operate on a price `pd.Series` and return a transformed
series; for OHLCV models the transform scales OHLC consistently with
`close`.

| Scenario | Transform |
|---|---|
| **Flash crash** | Inject an N% drop on a chosen date, with optional partial recovery over M bars. |
| **Volatility scaling** | Demean log-returns, multiply by factor *k*, rebuild the path (trend preserved). |
| **Trend / drift** | Add a compounding annual drift (bull or bear) on top of the real path. |
| **Return shuffle (MCPT)** | log-returns → seeded shuffle of order → cumulative rebuild. The "is the edge real?" diagnostic. |

Public surface:
- `Scenario` dataclass: `kind` (enum/str), parameters, and `seed`.
- `apply(prices: pd.Series) -> pd.Series` (deterministic given the seed).
- A preset registry the sidebar enumerates (label + default params).

Date-window presets (`app/presets.py`) are extended with 2008 GFC,
2010-05-06 flash crash, and 2018 Q4 windows. These remain distinct from
the scenario engine (they select *real* windows; the engine *warps*
data).

### 3. Visualizations — `app/ui/analytics_view.py`

All four panels derive from the existing `result.equity_curve` and
`result.position`, so **no model changes are required** — the
lowest-risk workstream.

- **Rolling Sharpe** line + **returns histogram** (strategy vs buy-and-hold).
- **Monthly returns heatmap** (year × month grid).
- **Trade table & stats** — win rate, avg win/loss, profit factor, durations.
- **Exposure / position timeline** — shaded long/short/flat plus a
  position-size sub-panel.

**Targeted refactor:** the trade-extraction logic currently inline in
`app/ui/data_view.py::_trade_markers` is promoted to a shared
`tradinglib/eval/trades.py::trades_from_position(position, prices)`
returning a trades DataFrame (entry, exit, side, pnl, duration). Both
the price-chart markers and the new trade table consume it, removing
duplication.

### 4. Backtest controls + comparison toggle

The sidebar gains a collapsible **"Advanced"** section:
- Execution costs: fee (bps), slippage (bps), initial capital.
- Position sizing: fixed-fraction toggle + a size multiplier.
- The dynamic per-model parameter widgets from §1.

**Baseline comparison** — a "compare against baseline" checkbox. When
enabled (or whenever a scenario / non-default ticker is active), the app
runs the configured backtest a second time **without** the scenario
transform and overlays "stressed vs baseline" equity curves so the
impact is visible. Implemented by calling `app.adapters.run` twice; no
engine changes.

---

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `model.md` frontmatter | Declares tickers, params, scenario capability | — |
| `tradinglib/scenarios/` | Pure price-series transforms + presets | pandas, numpy |
| `tradinglib/eval/trades.py` | `trades_from_position` → trades DataFrame | pandas |
| Each model's `run_for_gui` | Standardized load→scenario→signal→backtest | scenarios, backtest engine |
| `app/adapters.py` | Assemble config dict, forward to model (double-run for compare) | run_for_gui contract |
| `app/streamlit_app.py` sidebar | Render dynamic widgets from metadata | models metadata |
| `app/ui/analytics_view.py` | New result-derived charts | result object, trades.py |

## Error handling

- Free-text ticker with no data → catch loader failure, surface a clear
  "no data for `<ticker>`" message (existing `FileNotFoundError`/`ValueError`
  handling in `streamlit_app.py` is extended).
- Scenario on a `scenario_capable: false` model → the scenario controls
  are hidden for that model (not merely disabled), so the state is
  unreachable.
- Invalid param combinations (e.g. fast ≥ slow) → validated before run
  with an inline error, matching the existing `start >= end` guard.

## Testing

- `tradinglib/scenarios/` — unit tests per transform: shape/index
  preservation, determinism under fixed seed, known-input/known-output
  for crash and drift, distribution preservation for shuffle.
- `tradinglib/eval/trades.py` — unit tests on a synthetic position
  series (round trips, open-at-end, flat-throughout).
- Each model's `run_for_gui` — smoke test with the new kwargs (default
  symbol, a scenario, custom costs) returns the expected dict shape.
- Existing tests continue to pass unchanged (contract is additive).

## Implementation order (phased workstreams)

1. **Foundation** — frontmatter metadata + standardized `run_for_gui`
   across all four models. *(Everything depends on this.)*
2. **Visualizations** — independent; derives from existing results.
3. **Ticker switching** — dynamic sidebar from metadata.
4. **Backtest controls** — costs / sizing / params.
5. **Scenario engine** — `tradinglib/scenarios/` + wiring.
6. **Comparison toggle** — double-run overlay.
