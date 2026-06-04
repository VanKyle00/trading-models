# Deploying the FastAPI workbench

The workbench (`webapp.main:app`) is a FastAPI app serving backtests over
`tradinglib.service` plus a bounded LLM chat console. The **primary** target is
[Modal](#modal-primary); a `Dockerfile` is also provided for container hosts
([Render](#render-or-any-container-host) / Railway / Fly).

## What's in the repo

| File | Purpose |
| --- | --- |
| `deploy/modal_app.py` | Modal app: builds the image from `pyproject.toml`, mounts a persistent Volume for the data cache, serves the ASGI app. |
| `Dockerfile` | Builds an equivalent image (uv + Python 3.12) for any container host. |
| `render.yaml` | Render Blueprint: a free Docker web service with a `/healthz` health check. |
| `.dockerignore` | Keeps local data caches, notebooks, and tests out of the image. |

## Modal (primary)

Modal builds and caches the image (no serverless size limit), serves the ASGI
app via `@modal.asgi_app()` with `@modal.concurrent` so one container handles
many requests and SSE streams, and persists the market-data cache in a Volume.

```bash
uv sync --extra deploy                      # installs the modal CLI
uv run modal token new                      # authenticate this machine (one-time)

# Create the secret holding your Anthropic key (one-time):
uv run modal secret create trading-models-secrets ANTHROPIC_API_KEY=sk-ant-...

# Deploy (re-run any time to ship changes):
uv run modal deploy deploy/modal_app.py
```

Modal prints the public URL on success; the app is at `/`. Notes:

- **Data cache.** `deploy/modal_app.py` mounts a `trading-models-data` Volume at
  `/app/data`, so downloaded parquet survives deploys. Modal background-commits
  the Volume; if a write isn't committed before a cold scaledown, the loader
  just re-fetches from yfinance (the graceful fallback).
- **Cold starts.** The function scales to zero (`min_containers=0`) and keeps a
  warm container for 5 min after the last request (`scaledown_window=300`). Set
  `min_containers=1` in `deploy/modal_app.py` to eliminate cold starts (costs
  more — a container stays resident).
- **No key?** Remove the `secrets=[...]` line to deploy without `ANTHROPIC_API_KEY`;
  the chat console then streams a graceful "Assistant is unavailable" event.

## Render (or any container host)

## One-time setup on Render

1. Push `main` to GitHub (already done) so Render can see the repo.
2. In the Render dashboard: **New → Blueprint**.
3. Select the `trading-models` repo. Render reads `render.yaml` and proposes the
   `trading-models-workbench` web service.
4. Under the service's **Environment**, set the secret:
   - `ANTHROPIC_API_KEY` = your Anthropic key.
   - (Optional) `ASSISTANT_MODEL` is preset to `claude-haiku-4-5`; override to
     `claude-sonnet-4-6` for stronger answers at higher cost.
5. **Apply** / **Create**. The first build takes a few minutes (it compiles the
   data/ML stack). When it's live, open the service URL — the app is at `/`.

After this, every push to `main` redeploys automatically.

## How data works on the host

Backtests load market data through `tradinglib.loaders`, which **downloads from
yfinance/Binance on first use and caches a parquet** under `data/processed/`. So
a fresh host needs no bundled data — the first backtest per symbol fetches live
(a few seconds), then subsequent runs hit the cache.

On Render's **free** plan the filesystem is writable but **ephemeral** — the
cache resets on each redeploy or restart, so the first run after a deploy is
slow again. To keep it warm, attach a **persistent disk** mounted at
`/app/data` (a paid feature):

```yaml
    disk:
      name: data-cache
      mountPath: /app/data
      sizeGB: 1
```

## Notes & limits

- **Free-tier cold starts.** Free web services spin down after ~15 min idle and
  cold-start on the next request (~30s). Fine for a demo; upgrade the plan to
  keep it always-on.
- **Chat without a key.** If `ANTHROPIC_API_KEY` is unset, the console still
  loads and streams a graceful "Assistant is unavailable right now." final
  event — the rest of the app is unaffected.
- **The chat is bounded.** It can only list models, read a model spec, and run
  backtests, with per-session token/run caps and per-IP rate limiting — no code
  execution.
- **Health check.** Render polls `/healthz`, which returns `{"status": "ok"}`.

## Local parity

The image's start command mirrors local dev:

```bash
# local
uv run uvicorn webapp.main:app --reload
# container (what Render runs)
.venv/bin/python -m uvicorn webapp.main:app --host 0.0.0.0 --port $PORT
```
