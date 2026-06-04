# Container image for the FastAPI workbench (webapp.main:app).
# Render builds this on push; the start command is the CMD below.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    PYTHONPATH=/app

# libgomp1 is the OpenMP runtime xgboost/scikit-learn link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for layer caching (project itself installed later).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# App code, then install the project (tradinglib) into the venv.
COPY . .
RUN uv sync --frozen --no-dev \
    && mkdir -p data/processed data/raw

# yfinance/binance loaders cache parquet here at runtime. On Render's free
# plan this filesystem is writable but ephemeral (resets on redeploy); attach
# a persistent disk mounted at /app/data to keep the cache warm across restarts.
EXPOSE 10000
CMD ["sh", "-c", ".venv/bin/python -m uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
