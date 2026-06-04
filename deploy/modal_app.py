"""Deploy the FastAPI workbench (``webapp.main:app``) to Modal.

    uv run modal deploy deploy/modal_app.py

Prerequisites (one-time):

    uv run modal token new                       # authenticate this machine
    uv run modal secret create trading-models-secrets ANTHROPIC_API_KEY=sk-ant-...

The image installs the project's locked dependencies and ships the app code.
The yfinance/Binance parquet cache lives in a Modal Volume mounted at
``/app/data`` so backtests stay warm across deploys and restarts (Modal
background-commits the Volume; if a write isn't committed before a cold
scaledown the loader simply re-fetches — the graceful fallback).
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent  # repo root (has pyproject.toml)

# Files that must NOT be uploaded with the app: local data caches, the venv,
# git, notebooks, and assorted caches. Keeps the bundle small.
_IGNORE = [
    ".git/**",
    ".venv/**",
    # ALL of data/ — the Volume mounts at /app/data and Modal refuses to mount
    # over a non-empty path, so the image must not ship any data/ files.
    "data/**",
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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")  # OpenMP runtime that xgboost / scikit-learn link against
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))  # [project.dependencies]
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})  # so `import webapp` resolves
    .add_local_dir(str(ROOT), remote_path="/app", ignore=_IGNORE)
)

app = modal.App("trading-models-workbench")

# Persistent cache for downloaded market data. processed_dir() resolves to
# /app/data/processed, which this Volume backs.
data_volume = modal.Volume.from_name("trading-models-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/app/data": data_volume},
    # Holds ANTHROPIC_API_KEY. Remove this line to deploy without the key — the
    # chat console then degrades gracefully to "Assistant is unavailable".
    secrets=[modal.Secret.from_name("trading-models-secrets")],
    min_containers=0,  # scale to zero; set to 1 to eliminate cold starts (costs more)
    scaledown_window=300,  # keep a warm container for 5 min after the last request
    timeout=300,
)
@modal.concurrent(max_inputs=100)  # one container serves many requests + SSE streams
@modal.asgi_app()
def fastapi_app():
    from webapp.main import app as web_app

    return web_app
