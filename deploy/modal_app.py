"""Deploy the FastAPI workbench (``webapp.main:app``) to Modal.

    uv run modal deploy deploy/modal_app.py

Prerequisites (one-time):

    uv run modal token new                       # authenticate this machine
    uv run modal secret create trading-models-secrets ANTHROPIC_API_KEY=sk-ant-...

Two services share one Modal app:

* ``fastapi_app`` — the CPU web container. Serves the workbench and the
  assistant console. Cheap; scales to zero. It does NOT load any local model.
* ``LocalModel`` — a GPU container that loads base Qwen2.5-7B + our fine-tuned
  LoRA adapter once and answers ``complete`` calls. It only wakes when a user
  picks "Our model" in the console and scales back to zero after idle, so the
  GPU is billed only during actual local-model use. The web container reaches
  it remotely (``ASSISTANT_LOCAL_BACKEND=modal`` -> ``ModalRemoteProvider``).

The image installs the project's locked dependencies and ships the app code.
The yfinance/Binance parquet cache lives in a Modal Volume mounted at
``/app/data`` so backtests stay warm across deploys and restarts (Modal
background-commits the Volume; if a write isn't committed before a cold
scaledown the loader simply re-fetches — the graceful fallback).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent  # repo root (has pyproject.toml)

# The adapter "Our model" serves. 165 MB LoRA on top of Qwen2.5-7B-Instruct.
# Kept in sync with webapp.main._ASSISTANT_ADAPTER via the env var set on the
# GPU image below.
ADAPTER = "qwen25-7b-assistant-n21-ep3"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Files that must NOT be uploaded with the app: local data caches, the venv,
# git, notebooks, and assorted caches. Keeps the bundle small.
_IGNORE = [
    ".git/**",
    ".venv/**",
    # ALL of data/ — the Volume mounts at /app/data and Modal refuses to mount
    # over a non-empty path, so the image must not ship any data/ files.
    "data/**",
    # All adapters (each ~165 MB). The CPU web image needs none; the GPU image
    # adds back only the one it serves (see gpu_image below).
    "adapters/**",
    "notebooks/**",
    "**/__pycache__/**",
    "*.pyc",
    ".superpowers/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    "tests/**",
    "*.png",
]


def _download_base_model() -> None:
    """Bake the base weights into the GPU image so cold starts don't download
    ~15 GB on the first request (which would blow the web request timeout)."""
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL)


# Deps + workdir, NO local code yet. Modal forbids build steps (pip_install,
# run_function) AFTER add_local_*, so the shared base stops before adding code;
# each derived image runs its build steps, then adds local files LAST.
base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")  # OpenMP runtime that xgboost / scikit-learn link against
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))  # [project.dependencies]
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})  # so `import webapp` resolves
)

# Web image: CPU-only. Tell the app to reach the local model over Modal instead
# of loading it in-process (the in-process path is for WSL2 / local GPU).
web_image = (
    base.env({"ASSISTANT_LOCAL_BACKEND": "modal"}).add_local_dir(
        str(ROOT), remote_path="/app", ignore=_IGNORE
    )  # last
)

# GPU image: inference stack + baked base weights (all build steps first),
# then app code + the one adapter added last.
gpu_image = (
    base.pip_install(
        "torch>=2.4",
        "transformers>=4.46,<5",  # pin: the model was trained on a 4.x stack; 5.x is untested here
        "peft>=0.13",
        "accelerate>=1.0",
        "bitsandbytes>=0.44",  # 4-bit (nf4) load; Ampere+ GPU required
        "huggingface_hub>=0.25",
    )
    .env({"ASSISTANT_ADAPTER": f"adapters/{ADAPTER}"})
    .run_function(_download_base_model)  # bake base weights into the image layer
    .add_local_dir(str(ROOT), remote_path="/app", ignore=_IGNORE)  # code (no adapters)
    .add_local_dir(  # the one adapter we serve, added back
        str(ROOT / "adapters" / ADAPTER),
        remote_path=f"/app/adapters/{ADAPTER}",
    )
)

app = modal.App("trading-models-workbench")

# Persistent cache for downloaded market data. processed_dir() resolves to
# /app/data/processed, which this Volume backs.
data_volume = modal.Volume.from_name("trading-models-data", create_if_missing=True)


@app.cls(
    image=gpu_image,
    gpu="A10G",  # 24 GB; comfortable for 4-bit 7B. Swap for "L4" (cheaper) or "A100" (faster).
    scaledown_window=300,  # keep the GPU warm 5 min after the last call, then scale to zero
    timeout=600,  # generation budget per call (multi-turn tool loops re-enter complete)
)
class LocalModel:
    """GPU container serving our fine-tuned model. One generation at a time
    (Modal default: 1 input per container); Modal adds containers under load."""

    @modal.enter()
    def _load(self) -> None:
        import os

        from tradinglib.assistant.local_provider import LocalAdapterProvider

        adapter = os.environ.get("ASSISTANT_ADAPTER", f"adapters/{ADAPTER}")
        self._provider = LocalAdapterProvider(adapter_path=adapter)

    @modal.method()
    def complete(self, system, conversation, tools):
        return self._provider.complete(system, conversation, tools)


@app.function(
    image=web_image,
    volumes={"/app/data": data_volume},
    # Holds ANTHROPIC_API_KEY. Remove this line to deploy without the key — the
    # chat console then degrades gracefully to "Assistant is unavailable".
    secrets=[modal.Secret.from_name("trading-models-secrets")],
    min_containers=0,  # scale to zero; set to 1 to eliminate cold starts (costs more)
    scaledown_window=300,  # keep a warm container for 5 min after the last request
    timeout=600,  # allows a GPU cold start (container spin + first generation) within one request
)
@modal.concurrent(max_inputs=100)  # one container serves many requests + SSE streams
@modal.asgi_app()
def fastapi_app():
    from webapp.main import app as web_app

    # Volumes are snapshots: a warm container keeps seeing the volume as it was
    # when it mounted, so a report committed by scheduled_swing_scan would stay
    # invisible until the container recycled. Reload before serving /scans and
    # /tournaments (both read the volume-backed reports + ledger).
    @web_app.middleware("http")
    async def _reload_scans_volume(request, call_next):
        if request.url.path.startswith(("/scans", "/tournaments")):
            # a stale read beats a 500; the next container recycle catches up
            with contextlib.suppress(Exception):
                await data_volume.reload.aio()
        return await call_next(request)

    return web_app


@app.function(
    image=web_image,
    volumes={"/app/data": data_volume},
    secrets=[modal.Secret.from_name("trading-models-secrets")],  # LLM briefs need the key
    # 22:00 UTC weekdays = 17:00/18:00 ET — after the US close year-round, so
    # the scan sees the day's final bars. The report lands on the shared data
    # Volume, which the /scans page in fastapi_app reads.
    schedule=modal.Cron("0 22 * * 1-5"),
    # ~1000 .info fetches + ~160 EDGAR companyfacts + ~80 walk-forward tournaments
    # + chains for survivors + ~15 LLM briefs. The 2026-06-10 live smoke ran in
    # ~2.5 min on a warm cache; a cold Russell-scale night is unmeasured, and a
    # timeout kill costs the whole nightly report — so take the headroom.
    timeout=7200,
)
def scheduled_swing_scan() -> None:
    from datetime import UTC, datetime, timedelta

    from tradinglib.assistant.provider import ClaudeProvider
    from tradinglib.data.paths import processed_dir
    from tradinglib.loaders.equities.yfinance import corp_actions_since
    from tradinglib.scanner.config import ScanConfig
    from tradinglib.scanner.ledger import build_ledger, load_ledger, write_ledger
    from tradinglib.scanner.pipeline import run_scan
    from tradinglib.scanner.report import load_latest_report, write_report

    base = processed_dir("scans")
    # without last night's report every winner_changed stays null forever
    previous = load_latest_report(base, before=datetime.now(UTC).strftime("%Y-%m-%d"))

    certification = None
    cert_path = base / "certification.json"
    try:
        import json

        import pandas as pd

        from tradinglib.loaders.equities.yfinance import load_daily
        from tradinglib.loaders.events.earnings import get_earnings_dates
        from tradinglib.scanner.pooled import build_certification

        if cert_path.exists():
            certification = json.loads(cert_path.read_text(encoding="utf-8"))
            # valid JSON, wrong shape (e.g. verdicts a list) would blow up inside
            # run_scan, outside this try/except — treat malformed as missing so the
            # stale-rebuild path below kicks in.
            if not (
                isinstance(certification, dict) and isinstance(certification.get("verdicts"), dict)
            ):
                certification = None
        built = (certification or {}).get("built_asof")
        stale = (
            built is None
            or (datetime.now(UTC) - datetime.strptime(built, "%Y-%m-%d").replace(tzinfo=UTC)).days
            >= 7
        )
        if stale and previous:
            # certification family = last night's archived gate output; bars deep
            # enough for the sweep's oldest night (lookback + setup window)
            config = ScanConfig()
            tickers = sorted(
                {r["ticker"] for st in ("long", "short") for r in previous["fa_candidates"][st]}
            )
            start = (
                datetime.now(UTC) - timedelta(days=config.pooled_lookback_days + 450)
            ).strftime("%Y-%m-%d")
            bars_by_ticker: dict = {}
            earnings_by_ticker: dict = {}
            for ticker in tickers:
                try:
                    bars_by_ticker[ticker] = load_daily(ticker, start=start)
                    earnings = get_earnings_dates([ticker])
                    dts = pd.DatetimeIndex(earnings["earnings_datetime"])
                    if dts.tz is None:
                        dts = dts.tz_localize("UTC")
                    # tz-naive to match the sweep's internally naive-ized bars
                    # (detect_pead compares these against the bars index directly)
                    earnings_by_ticker[ticker] = dts.tz_convert("UTC").tz_localize(None)
                except Exception:
                    earnings_by_ticker.setdefault(ticker, pd.DatetimeIndex([]))
            print(f"certification bars: {len(bars_by_ticker)}/{len(tickers)} tickers loaded")
            certification = build_certification(
                bars_by_ticker,
                earnings_by_ticker,
                asof=pd.Timestamp(datetime.now(UTC).strftime("%Y-%m-%d")),
                config=config,
            )
            cert_path.write_text(json.dumps(certification, indent=2), encoding="utf-8")
            print(
                f"certification rebuilt: {sum(1 for v in certification['verdicts'].values() if v['certified'])}"
                f"/{len(certification['verdicts'])} certified, sim_errors={certification['sim_errors']}"
            )
    except Exception as exc:
        # must never cost the night's report; a prior (possibly stale) sidecar still applies
        print(f"certification refresh failed (scan proceeds): {exc}")

    result = run_scan(
        ScanConfig(),
        ClaudeProvider(),
        previous_report=previous,
        # yesterday's rebuild: a campaign that exited TODAY still suppresses tonight (deliberate one-night lag; report must never wait on ledger work)
        recent_ledger=load_ledger(base),
        certification=certification,
    )
    json_path, _ = write_report(result, base / result["asof"])
    try:
        # forward ticket-performance ledger; must never cost the night's report
        ledger = build_ledger(base, asof=result["asof"], actions_probe=corp_actions_since)
        write_ledger(ledger, base)
        print(f"ledger {result['asof']}: {ledger['stats']}")
    except Exception as exc:
        print(f"ledger build failed (report still written): {exc}")
    data_volume.commit()  # persist before the container scales down
    print(f"swing scan {result['asof']}: {result['funnel']} -> {json_path}")
