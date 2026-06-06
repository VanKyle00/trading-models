"""FastAPI app: a server-rendered workbench over tradinglib.service.

Run locally with::

    uv run uvicorn webapp.main:app --reload

``app`` is the ASGI application; ``create_app()`` builds a fresh instance
(used by tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from tradinglib.assistant import Budget, RateLimiter, run_chat
from tradinglib.assistant import provider as _assistant_provider
from tradinglib.service import RequestError, list_specs, model_spec, run, run_to_dict
from webapp.charts import build_all
from webapp.events import events_for_assets
from webapp.forms import request_from_payload

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_CHAT_LIMITER = RateLimiter(max_per_window=30)

# The assistant console can run on the hosted Claude teacher ("claude") or our
# fine-tuned LoRA adapter ("local"). The adapter path is env-overridable; the
# 7B load is slow and needs a GPU, so it's loaded once and cached per process.
_ASSISTANT_ADAPTER = os.environ.get("ASSISTANT_ADAPTER", "adapters/qwen25-7b-assistant-n20-ep3")
_local_provider_cache: dict[str, Any] = {}


def _get_local_provider() -> Any:
    provider = _local_provider_cache.get(_ASSISTANT_ADAPTER)
    if provider is None:
        from tradinglib.assistant.local_provider import LocalAdapterProvider

        provider = LocalAdapterProvider(adapter_path=_ASSISTANT_ADAPTER)
        _local_provider_cache[_ASSISTANT_ADAPTER] = provider
    return provider


# (key, label, kind, tone) — kind drives formatting, tone drives colour.
#   kind: "pct" → percent, "ratio" → 2dp, "int" → integer
#   tone: "signed" → green/red by sign, "down" → always red, "neutral" → ink
_HERO_METRICS = [
    ("sharpe", "Sharpe", "ratio", "signed"),
    ("annualized_return", "Ann. Return", "pct", "signed"),
    ("max_drawdown", "Max Drawdown", "pct", "down"),
    ("hit_rate", "Hit Rate", "pct", "neutral"),
]
_SECONDARY_METRICS = [
    ("sortino", "Sortino", "ratio"),
    ("probabilistic_sharpe", "Prob. Sharpe", "pct"),
    ("deflated_sharpe", "Deflated Sharpe", "pct"),
    ("n_bars", "Bars", "int"),
]


def _fmt(value: Any, kind: str, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if kind == "pct":
        s = f"{value * 100:.1f}%"
        return f"+{s}" if signed and value > 0 else s
    if kind == "int":
        return f"{int(value)}"
    s = f"{value:.2f}"
    return f"+{s}" if signed and value > 0 else s


def _tone(value: Any, tone: str) -> str:
    if tone == "down":
        return "down"
    if tone == "signed" and isinstance(value, (int, float)):
        return "up" if value > 0 else "down" if value < 0 else "neutral"
    return "neutral"


def _metric_view(metrics: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hero = [
        {
            "label": label,
            "value": _fmt(metrics[k], kind, tone == "signed"),
            "tone": _tone(metrics[k], tone),
        }
        for k, label, kind, tone in _HERO_METRICS
        if k in metrics
    ]
    secondary = [
        {"label": label, "value": _fmt(metrics[k], kind)}
        for k, label, kind in _SECONDARY_METRICS
        if k in metrics
    ]
    return hero, secondary


def _chat_context(raw: Any) -> str | None:
    """Format the frontend's on-screen run descriptor into a plain-text summary.

    Expects a dict like ``{"model": ..., "symbol": ..., "start": ..., "end": ...,
    "metrics": {name: value}}``. Returns ``None`` for anything malformed so the
    assistant simply falls back to standalone mode.
    """
    if not isinstance(raw, dict):
        return None
    lines: list[str] = []
    head = " · ".join(str(raw[k]) for k in ("model", "symbol") if raw.get(k) not in (None, ""))
    window = " to ".join(str(raw[k]) for k in ("start", "end") if raw.get(k) not in (None, ""))
    if head:
        lines.append(f"Backtest: {head}" + (f" ({window})" if window else ""))
    metrics = raw.get("metrics")
    if isinstance(metrics, dict) and metrics:
        rendered = ", ".join(f"{name}={value}" for name, value in metrics.items())
        lines.append(f"Metrics: {rendered}")
    return "\n".join(lines) or None


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
        specs = list_specs()
        events_by_model = {s.id: events_for_assets(s.assets) for s in specs}
        return _TEMPLATES.TemplateResponse(
            request, "index.html", {"specs": specs, "events_by_model": events_by_model}
        )

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
        hero, secondary = _metric_view(br.result.metrics)
        # Compact descriptor the assistant console echoes back with each chat
        # message, so the agent answers against the run currently on screen.
        run_context = {
            "model": spec.name,
            "symbol": br.symbol,
            "start": form.get("start"),
            "end": form.get("end"),
            "metrics": {c["label"]: c["value"] for c in hero},
        }
        return _TEMPLATES.TemplateResponse(
            request,
            "_results.html",
            {
                "symbol": br.symbol,
                "model_name": spec.name,
                "hero": hero,
                "secondary": secondary,
                "figures": figures,
                "run_context_json": json.dumps(run_context),
            },
        )

    @app.post("/api/v1/chat", response_model=None)
    async def chat(request: Request) -> StreamingResponse | JSONResponse:
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        message = str(payload.get("message", "")).strip()
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        context = _chat_context(payload.get("context"))
        # "claude" (hosted teacher) or "local" (our fine-tuned LoRA adapter).
        use_local = str(payload.get("provider", "claude")).lower() == "local"

        client_ip = request.client.host if request.client else "unknown"
        if not _CHAT_LIMITER.allow(client_ip):
            return JSONResponse({"error": "rate limit reached, try later"}, status_code=429)

        def stream() -> Any:
            try:
                if use_local:
                    try:
                        provider = _get_local_provider()
                    except Exception:  # no GPU / adapter missing -> tell the user, don't 500
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "final",
                                    "text": "Local model unavailable (needs a GPU and a "
                                    "trained adapter). Switch to Claude.",
                                }
                            )
                            + "\n\n"
                        )
                        return
                else:
                    provider = _assistant_provider.ClaudeProvider()
                for event in run_chat(message, provider, Budget(), context=context):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception:  # never break the SSE stream; always end with a final event
                yield (
                    "data: "
                    + json.dumps({"type": "final", "text": "Assistant is unavailable right now."})
                    + "\n\n"
                )

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
