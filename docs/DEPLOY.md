# Deploying the FastAPI workbench

The workbench (`webapp.main:app`) is a containerised FastAPI app. It serves
backtests over `tradinglib.service` and a bounded LLM chat console. This guide
covers [Render](https://render.com); any container host works with the same
`Dockerfile`.

## What's in the repo

| File | Purpose |
| --- | --- |
| `Dockerfile` | Builds the image (uv + Python 3.12, installs locked deps, runs uvicorn). |
| `render.yaml` | Render Blueprint: a free Docker web service with a `/healthz` health check. |
| `.dockerignore` | Keeps local data caches, notebooks, and tests out of the image. |

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
