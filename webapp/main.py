"""FastAPI app: a server-rendered workbench over tradinglib.service.

Run locally with::

    uv run uvicorn webapp.main:app --reload

``app`` is the ASGI application; ``create_app()`` builds a fresh instance
(used by tests).
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Trading Models — Workbench")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
