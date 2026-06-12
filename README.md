# Trading Models

📄 **[Read the research →](https://vankyle00.github.io/trading-models/docs/index.html)** — two tracks: in-depth **model breakdowns** (one self-contained paper per model) and first-principles **concept writeups** (the theory behind the models).

🛠 **[Engineering devlog →](https://vankyle00.github.io/trading-models/docs/devlog.html)** — the build history, newest first: why each major change landed, what broke along the way, and what fixed it.

A growing portfolio of trading models spanning classical quant, machine
learning, market microstructure, and alternative data — US equities and
crypto. Each model lives in a self-contained directory under
`models/<family>/` with a reproducible notebook, a `backtest.py` entry point,
and standardized metrics; the shared `tradinglib/` engine measures every
model the same way, so the results below are directly comparable.

## Key features

- **Unified backtest engine** (`tradinglib.backtest`) — one vectorized core
  every model runs through. Signals lag one bar and fill at the **next bar's
  open** (no look-ahead), with linear bps transaction costs; an event-driven
  front-end and a dedicated options engine (Greeks, multi-leg payoffs) feed
  the same core.
- **Standardized metrics** — annualized return, Sharpe, Sortino, max drawdown,
  hit rate, and turnover on every model; assumptions documented in
  [`docs/methodology.md`](docs/methodology.md).
- **Negative results are first-class** — hypotheses that the data rejects ship
  with the same rigor as the winners, inverse direction included.
- **Nightly scanner with a forward ledger** — a Russell-1000 funnel issues
  walk-forward-validated trade tickets and re-scores every ticket against
  subsequent bars, so the pipeline grades itself out of sample.
- **Our own trained assistant model** — the workbench's chat assistant runs on
  a provider abstraction that swaps between the Anthropic API and a self-hosted
  **Qwen2.5-7B** fine-tuned in-house (QLoRA, see below).
- **Live, deployed workbench** — themed FastAPI UI with Plotly charts,
  market-event presets, and the grounded LLM console, running on Modal.

## Live demo

**▶ [Open the workbench →](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run)**
— the interactive test area, deployed on [Modal](https://modal.com). Pick a
model, jump straight to a notable **market event** (COVID crash, 2022 bear,
GFC 2008, FTX collapse, … — the list adapts to the model's asset class), and
run a backtest over any window. Results render as rich Plotly charts with a
hero metric strip, and a built-in **LLM assistant** answers questions grounded
in the run you're looking at. Bone / night themes, nothing to install.

> First load can take a few seconds — the app scales to zero when idle.

[![The Trading Models workbench (night theme): a delta-hedged options backtest over the 2008 financial crisis, with the bounded LLM assistant analysing the result](docs/assets/workbench.png)](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run)

There's also the original **[Streamlit app](https://trading-models-swqny2mhsqftylrq8hj3w9.streamlit.app/)**,
which serves the same backtests via the shared `tradinglib.service` layer.

### Run the workbench locally

```bash
uv sync
uv run uvicorn webapp.main:app --reload    # FastAPI workbench → http://localhost:8000
uv run streamlit run app/streamlit_app.py  # original Streamlit app
```

The chat assistant uses the Anthropic API — set `ANTHROPIC_API_KEY` in the
environment to enable `/api/v1/chat`. The model defaults to Claude Haiku 4.5;
override with `ASSISTANT_MODEL` (e.g. `claude-sonnet-4-6`). The assistant is a
bounded agent: it can only list models, read a model's spec, and run backtests —
no code execution — with per-session token/run caps and per-IP rate limiting.

The **options planner** (`/planner`, also available from the index console)
turns a stated hypothesis — directional or range-bound ("I'm bullish on
RIVN") — into a priced options ticket. It proposes ATR- and
structure-grounded entry/stop/target scenarios on a chart card (a price band
for a neutral view), warns about upcoming earnings and ex-dividend dates,
then prices structures against the live option chain — long option, debit
spread, cash-secured put, credit spread, or an iron condor / iron butterfly —
and presents one sized, liquidity-gated recommendation as a structured card:
legs, max loss/gain, breakeven, market-implied PoP, and a prefilled
OptionStrat payoff link. Sizing defaults to $100,000 and 1% risk (one bundled
confirmation — just say "go"). Outside market hours, when Yahoo zeroes option
bid/ask, it falls back to CBOE delayed quotes and warns that fills must be
re-verified at the open. Every number comes from the strategist pipeline,
never the model; the conversation is held client-side and nothing is
persisted.

### Our own trained model

The assistant is built on an `LLMProvider` protocol (`tradinglib/assistant/`),
so the agent loop never depends on a specific vendor. `ClaudeProvider` is the
default; `LocalAdapterProvider` serves a **self-hosted Qwen2.5-7B-Instruct**
fine-tuned in-house — both implement the same interface and drop in with no
changes to `agent.py` or `tools.py`.

The training track lives under `tradinglib/training/` and `scripts/`:

- **QLoRA fine-tune** — Qwen2.5-7B in 4-bit on a single 16 GB consumer GPU
  (RTX 5080, WSL2), `r=16`/`alpha=32` LoRA across all attention + MLP
  projections. Hyperparameters are pinned dataclasses in
  `tradinglib/training/config.py`.
- **Grounded SFT dataset** — built from real backtest traces
  (`scripts/build_dataset.py`, `tradinglib/dataset/`) so the model learns to
  ground every numeric claim in tool output, matching the bounded-agent
  contract.
- **Swap-in serving** — `LocalAdapterProvider` parses Qwen-style
  `<tool_call>` blocks and speaks the same neutral turn type the agent loop
  expects; heavy deps (torch/peft/bitsandbytes) are lazily imported so CI stays
  GPU-free.

Full runbook (install, smoke test, full run) is in
[`docs/training-assistant.md`](docs/training-assistant.md). Train with:

```bash
uv sync --extra train
uv run python scripts/build_dataset.py
uv run python scripts/train_assistant.py --train data/dataset/train.jsonl \
    --eval data/dataset/eval.jsonl --out adapters/qwen25-7b-assistant
```

### Deploy the workbench

Primary target is **[Modal](https://modal.com)** (`deploy/modal_app.py`):

    uv sync --extra deploy
    uv run modal token new
    uv run modal secret create trading-models-secrets ANTHROPIC_API_KEY=sk-ant-...
    uv run modal deploy deploy/modal_app.py

A `Dockerfile` (+ `render.yaml` blueprint) is also included for container hosts
like Render, Railway, or Fly. The chat degrades gracefully without the API key.
Full steps and the persistent-cache notes are in [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Nightly swing scanner

**▶ [Open the scans page →](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run/scans)**
— every weekday after the US close (22:00 UTC) a Modal cron sweeps the full
Russell 1000 (~1,000 names; S&P 500 available via `--universe sp500`) for swing setups on the 2-week-to-6-month horizon and publishes a
ranked watchlist to the workbench's `/scans` page: funnel stats up top, then
one card per candidate with the detected setup, trigger/stop levels, and a
grounded LLM brief.

[![A nightly scan report: the S&P 500 funnel (universe 503 → 40 past the FA gate → 4 setups forming) and the top-ranked candidate cards with trigger/stop levels and grounded LLM briefs](docs/assets/scans.png)](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run/scans)

The funnel (`tradinglib/scanner/`) narrows ~1,000 names to a handful in four
stages — the first two are the **fundamental (FA) gate**:

1. **FA gate, pass 1 — snapshot percentiles.** Six metrics — revenue growth,
   earnings growth, operating margin, debt-to-equity, free-cash-flow yield,
   and forward P/E — are scored as *cross-sectional percentiles* across the
   universe. Forward P/E is percentiled within its GICS sector, and the
   scoring is direction-aware: lower debt and a cheaper multiple score
   higher. A ticker's
   `fa_score` is the mean of the percentiles it actually has, with two hard
   filters: at least 4 of the 6 metrics present, and positive
   trailing-twelve-month revenue. The top 80 by `fa_score` advance.
2. **FA gate, pass 2 — EDGAR trend blend.** For each pass-1 survivor the
   scanner pulls quarterly XBRL companyfacts from SEC EDGAR and computes
   revenue YoY growth, revenue acceleration, and EPS change YoY. Those are
   percentiled among the survivors and blended as
   `0.7 · fa_score + 0.3 · edgar_score`; the re-ranked top 40 pass the gate.
   Tickers EDGAR has no data for keep their unblended score rather than being
   penalized for missing facts.
3. **Setup detection.** Three long/short detector pairs run over both FA
   cohorts (top-40 longs, bottom-40 shorts): `base_breakout`/`base_breakdown`
   (tight consolidation near the 52-week high or low on drying-up volume),
   `ma_pullback`/`ma_rally_fade` (orderly pullback to a rising 50-day MA or
   rally fade to a falling 50-day MA), and `pead`/`pead_down`
   (post-earnings-announcement drift after a big up- or down-gap on volume).
   Each emits a 0–1 score plus concrete trigger and stop levels.
4. **LLM document briefs + ranking.** Every finalist gets a bounded doc pack —
   the latest 8-K excerpt, the 10-Q/10-K MD&A opening, recent headlines, its
   FA metrics and the detected setup — and one LLM call returns strict JSON
   (thesis, catalysts, risks, red flags, stance, 0–10 qualitative score).
   Final rank is `0.35·FA + 0.45·setup + 0.20·qualitative`; an `avoid` stance
   or any red flag pins the name to the bottom of the list with the reason
   shown — never silently dropped. Candidates reporting earnings within 14
   days carry a warning chip.

The FA gate is two-sided, and the same nightly run also feeds a **strategy
tournament**: the top-N FA names become long candidates and the bottom-N
short candidates, each walk-forward tested (anchored 378/63-bar windows,
costs on) against a registry of 9 strategies across 29 parameter
configurations — the classic five (SMA crossover, Donchian breakout, RSI(2)
pullback, MACD, Bollinger fade), three setup strategies translated directly
from the scanner's own detectors (base breakout, MA pullback, PEAD), and
`ridge_momentum` (closed-form ridge regression on lagged returns, volatility,
and SMA distances, fit per walk-forward window on train bars only). A ticket
from a setup strategy means the setup has per-ticker walk-forward-validated
edge *and* fired tonight. Only survivors clear the bar — deflated-Sharpe
probability ≥ 0.90 corrected for *every* strategy and parameter tried on that
ticker, ≥ 12 OOS trades, stable parameters — and each winner becomes a
**trade ticket**: entry/stop/target from the winning rule, risk-based sizing,
and option structures (short-premium spreads, never naked calls) built from
the real chain behind a liquidity gate. Tickets render below the watchlist on
`/scans`; the strategy registry and the standalone models are documented at
[`/models`](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run/models).
To add a strategy see [docs/adding-a-model.md](docs/adding-a-model.md).
Quotes are indicative last/close marks: this is decision support that accrues
a forward paper-trading record, not an auto-trader.

Nightly output is two-tier: **tickets** clear the hard survival bar (DSR ≥ 0.90,
≥ 12 OOS trades, stable parameters) *and* a Benjamini-Hochberg FDR pass across
every ticker-stance run that night (α = 0.10); **watchlist** entries are labeled
demotions — survivors that cleared the hard bar but failed the nightly FDR —
with the demotion reason recorded alongside the candidate. The forward ledger
tracks both tiers identically, so the tiering itself is validated by out-of-sample
performance rather than assumed.

Issuance is disciplined and evidence-gated. A re-issue **cooldown** suppresses
a (ticker, stance, strategy, tier) campaign that is still waiting or open, so
a persistent setup is one campaign — not a new row every night. A weekly
**pooled certification** job pools each setup type's full cross-ticker
history and promotes its watch rows to tickets only at deflated-Sharpe ≥ 0.90
on ≥ 20 pooled dates plus an FDR pass across the type menu — a bar nothing
clears yet, deliberately. And the whole funnel is **replayable without
look-ahead** (`scripts/backfill_scan.py`), which is how we caught our own
+23.4R headline being ~3× inflated by re-fired campaigns; that story and the
rest of the build history are in the
[devlog](https://vankyle00.github.io/trading-models/docs/devlog.html).

That forward record is kept honest on the
[`/tournaments`](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run/tournaments)
page: each night's pipeline story (universe → FA gate → tournament verdicts →
tickets) is cataloged by date, and every ticket ever issued is re-scored
nightly by paper-trading its entry/stop/target levels against subsequent
daily bars — status, R-multiple, and price path vs levels, plus a cumulative
hit rate and total R. Entries fill per their trigger type within the row's entry window — 5 sessions
for most setups, 15 for the PEAD pair; stop-style triggers and protective stops
gap-fill at the open, never better than the plan; a bar that touches both stop
and target counts as stopped. Rebuild it locally with `uv run python scripts/evaluate_tickets.py`.

Run it yourself (`--limit` for a quick smoke run, `--skip-llm` to stop after
setup detection):

```bash
uv run python scripts/swing_scan.py --limit 25 --skip-llm
```

## Ticker sentiment

**▶ [Open the sentiment page →](https://van-kyle-00--trading-models-workbench-fastapi-app.modal.run/sentiment)**
— type a ticker and get the same story from three very different rooms: what
the press is printing, what investors are arguing on serious forums, and what
retail is shouting into the feeds. Each tier is scored independently — one
bounded LLM call per tier (strict JSON, the same grounded pattern as the
scanner briefs) on top of mechanical metrics computed in plain code — and the
page leads with the spread between them.

| Tier | Sources | Mechanical metrics (no LLM) |
| --- | --- | --- |
| **1 · Official media** | yfinance headlines + Google News RSS | headline count |
| **2 · Serious forums** | Seeking Alpha per-ticker RSS + r/stocks, r/investing, r/ValueInvesting, r/SecurityAnalysis | post count, mean upvotes/comments |
| **3 · Viral retail** | r/wallstreetbets + Stocktwits (user-tagged bull/bear) + Bluesky cashtag search + Google Trends | bull/bear ratio, WSB + Bluesky mentions, search-spike ratio (7d vs ~90d) |

- **The divergence callout is the point.** Overall bias is just the mean of
  the available tier scores; the interesting output is the banner that fires
  when two tiers disagree by ≥ 0.6 — viral froth the press hasn't blessed, or
  official optimism retail isn't buying, is exactly the read no single feed
  shows you.
- **Evidence can't be hallucinated.** The LLM cites pack-item *indices*; the
  server resolves them back to the real headlines and posts (links
  scheme-allowlisted), so every quote on a tier card is something that
  actually exists.
- **Free sources only, honest degradation.** X (~$200/mo) and TikTok (no
  usable API) are deliberately excluded — the viral tier is proxied by WSB +
  Stocktwits + Bluesky + Trends. Sources fail independently: a Cloudflare mood at
  Seeking Alpha or a missing Reddit key degrades that tier (status shown on
  the card), never the lookup, and an all-empty result renders "no data" —
  never a fake neutral. Reddit is the one keyed source (free script app via
  `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`; see
  [`docs/data-sources.md`](docs/data-sources.md)).
- **Cached per (ticker, day).** The first lookup of a day fetches live
  (~10–20 s); repeats are instant; **Re-fetch** bypasses. Finished reports
  accrue as JSON under `data/processed/sentiment/reports/` — the forward
  history a future nightly batch mode will reuse.

## Current models

| Model | Family | Window | Assets | OOS Sharpe | Max DD | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [SMA Crossover on SPY](https://vankyle00.github.io/trading-models/docs/models/01-sma-crossover-spy.html) | classical | swing | equities | 0.75 | -0.34 | working |
| [XGBoost Next-Day Return on SPY](https://vankyle00.github.io/trading-models/docs/models/02-xgboost-next-day-return-spy.html) | ml | swing | equities | 0.96 | -0.12 | working |
| [Google Trends Contrarian on BTC](https://vankyle00.github.io/trading-models/docs/models/04-google-trends-contrarian-btc.html) | alt-data | swing | crypto | -0.30 | -0.80 | negative-result |
| [Order Flow Imbalance on BTC](https://vankyle00.github.io/trading-models/docs/models/03-order-flow-imbalance-btc.html) | microstructure | intraday | crypto | -86.37 | -0.36 | negative-result |
| [Delta-Hedged Long Option on SPY](https://vankyle00.github.io/trading-models/docs/models/05-delta-hedged-long-option-spy.html) | options | swing | equities | -6.94 | -0.08 | working |
| [Earnings Event-Vol Straddle on SPY](https://vankyle00.github.io/trading-models/docs/models/06-earnings-straddle-spy.html) | options | swing | equities | 0.0 | 0.0 | negative-result |

Rows 3 and 4 are intentional negative results — hypotheses tested honestly,
rejected by the data, documented inverse direction included; the alternative
is a portfolio of overfit "winners". Row 5 posts a negative Sharpe by design:
it is the options-pipeline demonstrator, and its loss is the long-volatility
theta bleed the theory predicts. Row 6's thorough backtest (216 earnings
events across 9 names, 2020–2026) found **no statistically significant
edge** — the unfiltered straddle program bleeds (−$125.65/trade, p=0.052) and
the filtered branch's nominal gain (p=0.78) is an artifact of the synthetic
IV surface; its Sharpe/DD stay 0.0 because per-bar Sharpe is the wrong lens
for a sparse event trade. The microstructure Sharpe (−86.37) is annualized
from minute bars — the direction and the scale-invariant metrics (hit rate
29.7%, drawdown −36%) are what matter for comparison; see the model's README
for a daily-bar rescaling.

The full sortable index lives in [MODELS.md](MODELS.md) and is
auto-generated from each model's `model.md` frontmatter.

## Roadmap

- **Stance-aware ranking** — the nightly rank is cohort-blind, so short
  candidates (low FA scores by construction) sink in a mixed top-15; ranking
  within each cohort is the fix.
- **Surface the suppression evidence** — cooldown suppressions and the regime
  block are JSON-only today; they belong on the report pages.
- **Real-chain options backtests** — the planner already prices live chains;
  the options backtests still run on the synthetic surface. A historical
  chain loader is the upgrade.
- **Vendor-grade equities loaders** — Polygon or Alpaca to replace yfinance
  for production-quality bars (and retire its 429s at Russell scale).
- **L2-derived OFI** — the trade-side OFI experiment was a negative result;
  the next iteration uses depth-update events for a proper book-imbalance
  signal. Requires a WebSocket capture loader.

## Repository tour

| Directory | What lives there |
| --- | --- |
| `webapp/` | FastAPI workbench (the live demo) — themed UI, Plotly charts, market-event presets, LLM chat console |
| `app/` | Original Streamlit GUI for browsing models + running backtests |
| `deploy/` | `modal_app.py` — Modal deployment of the workbench + nightly cron (see [`docs/DEPLOY.md`](docs/DEPLOY.md)) |
| `tradinglib/` | Shared package — data, features, backtest engines, metrics, viz |
| `tradinglib/backtest/` | Vectorized + event-driven engines, options engine, standardized metrics |
| `tradinglib/scanner/` | The nightly funnel — FA gate, setup detectors, regime overlay, pooled certification |
| `tradinglib/tournament/` + `strategist/` | Walk-forward strategy tournament + ticket construction |
| `tradinglib/loaders/` | Data loaders, one subpackage per asset class |
| `tradinglib/assistant/` | Bounded LLM agent loop + provider abstraction (Claude / own Qwen adapter) |
| `tradinglib/sentiment/` | Three-tier ticker sentiment engine behind the `/sentiment` page |
| `tradinglib/training/` + `dataset/` | QLoRA fine-tuning glue + grounded SFT dataset builder |
| `data/ingestion/` | Documentation of each data source |
| `models/<family>/` | One directory per model — classical, ml, microstructure, options, alt-data |
| `docs/` | Research hub, [devlog](https://vankyle00.github.io/trading-models/docs/devlog.html), glossary, data sources, methodology, latency notes |
| `docs/models/` + `docs/concepts/` | Working papers per model + first-principles concept writeups |
| `scripts/` | Operational scripts — nightly scan, ticket evaluation, replay, training, index regeneration |
| `tests/` | Unit tests for `tradinglib`, the `webapp`, and the LLM assistant |

## Quick start

```bash
git clone https://github.com/<you>/trading-models.git
cd trading-models
uv sync --extra dev    # `uv sync` alone is enough if you only want to run the app
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

Want the theory behind a signal, not just its backtest? The
[**concept writeups**](https://vankyle00.github.io/trading-models/docs/concepts/index.html) develop the recurring
ideas from first principles — the first,
[*How Order Flow Shapes Liquidity*](https://vankyle00.github.io/trading-models/docs/concepts/01-order-flow-and-liquidity.html),
is the theory behind the microstructure model. Both tracks are reachable
from the [research index](https://vankyle00.github.io/trading-models/docs/index.html).

## Status

Six models live across all five families (classical, ML, microstructure,
options, alt-data), including intentional negative results. In production:
the shared backtest engine, the deployed workbench with its bounded LLM
assistant (Anthropic API or the own-trained Qwen2.5-7B provider), the nightly
Russell-1000 scanner with tournament tickets and the forward ledger that
grades them, and the three-tier sentiment page. The
[devlog](https://vankyle00.github.io/trading-models/docs/devlog.html) tracks
how it got here; the roadmap above is what's next.
