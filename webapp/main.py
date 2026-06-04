"""FastAPI app: a server-rendered workbench over tradinglib.service.

Run locally with::

    uv run uvicorn webapp.main:app --reload

``app`` is the ASGI application; ``create_app()`` builds a fresh instance
(used by tests).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from tradinglib.service import RequestError, list_specs, model_spec, run, run_to_dict
from webapp.charts import build_all
from webapp.forms import request_from_payload

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="Trading Models — Workbench")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/run")
    async def api_run(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                return JSONResponse(
                    {"error": "request body must be a JSON object"}, status_code=400
                )
            spec = model_spec(str(payload.get("model_id", "")))
            req = request_from_payload(payload, spec)
            result = run_to_dict(run(req))
        except (RequestError, KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "index.html", {"specs": list_specs()})

    @app.post("/run", response_class=HTMLResponse)
    async def run_partial(request: Request) -> HTMLResponse:
        form = dict(await request.form())
        try:
            spec = model_spec(str(form.get("model_id", "")))
            req = request_from_payload(form, spec)
            br = run(req)
        except (RequestError, KeyError, ValueError) as exc:
            return HTMLResponse(f'<p class="error">Could not run: {exc}</p>', status_code=400)
        figures = {
            name: fig.to_html(full_html=False, include_plotlyjs=False)
            for name, fig in build_all(br).items()
        }
        return _TEMPLATES.TemplateResponse(
            request,
            "_results.html",
            {
                "symbol": br.symbol,
                "model_name": spec.name,
                "metrics": br.result.metrics,
                "figures": figures,
            },
        )

    return app


app = create_app()
