"""FastAPI app: a server-rendered workbench over tradinglib.service.

Run locally with::

    uv run uvicorn webapp.main:app --reload

``app`` is the ASGI application; ``create_app()`` builds a fresh instance
(used by tests).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tradinglib.service import RequestError, model_spec, run, run_to_dict
from webapp.forms import request_from_payload


def create_app() -> FastAPI:
    app = FastAPI(title="Trading Models — Workbench")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/run")
    async def api_run(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            spec = model_spec(str(payload.get("model_id", "")))
            req = request_from_payload(payload, spec)
            result = run_to_dict(run(req))
        except (RequestError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    return app


app = create_app()
