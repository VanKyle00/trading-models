"""FastAPI app: a server-rendered workbench over tradinglib.service.

Run locally with::

    uv run uvicorn webapp.main:app --reload

``app`` is the ASGI application; ``create_app()`` builds a fresh instance
(used by tests).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from tradinglib.assistant import Budget, RateLimiter, run_chat
from tradinglib.assistant import provider as _assistant_provider
from tradinglib.service import RequestError, list_specs, model_spec, run, run_to_dict
from tradinglib.tournament.strategies import STRATEGIES
from webapp import scans as _scans
from webapp import sentiment as _sentiment
from webapp import tournaments as _tournaments
from webapp.charts import build_all
from webapp.events import events_for_assets
from webapp.forms import request_from_payload

logger = logging.getLogger(__name__)

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_CHAT_LIMITER = RateLimiter(max_per_window=30)
_HISTORY_MAX_MESSAGES = 20
_HISTORY_MAX_CHARS = 4_000

# The assistant console can run on the hosted Claude teacher ("claude") or our
# fine-tuned LoRA adapter ("local"). The adapter path is env-overridable; the
# 7B load is slow and needs a GPU, so it's loaded once and cached per process.
#
# ASSISTANT_LOCAL_BACKEND picks how "local" is served:
#   "inprocess" (default) — load the adapter in this process (WSL2 / local GPU).
#   "modal"               — call a remote Modal GPU container (set in the Modal
#                           deployment; see deploy/modal_app.py). Keeps the web
#                           container CPU-only so the GPU is billed only on use.
_ASSISTANT_ADAPTER = os.environ.get("ASSISTANT_ADAPTER", "adapters/qwen25-7b-assistant-n21-ep3")
_ASSISTANT_LOCAL_BACKEND = os.environ.get("ASSISTANT_LOCAL_BACKEND", "inprocess")
_local_provider_cache: dict[str, Any] = {}


def _get_local_provider() -> Any:
    provider = _local_provider_cache.get(_ASSISTANT_ADAPTER)
    if provider is None:
        if _ASSISTANT_LOCAL_BACKEND == "modal":
            from webapp.modal_provider import ModalRemoteProvider

            provider = ModalRemoteProvider()
        else:
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


def _fmt_money(value: Any, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    s = f"${abs(value):,.0f}"
    if value < 0:
        return f"-{s}"
    return f"+{s}" if signed and value > 0 else s


def _report_view(report: Any) -> dict[str, Any] | None:
    """Format the earnings-straddle synthetic report (``BacktestRun.extra['report']``)
    into display rows: the implied/expected move + k stats, and a filtered-vs-
    unfiltered equity/P&L comparison. Returns None for any model that emits no report.
    """
    if not isinstance(report, dict):
        return None
    fired = bool(report.get("filtered", {}).get("took_trade"))
    stats = [
        {"label": "Implied move", "value": _fmt(report.get("implied_move"), "pct")},
        {"label": "Expected move", "value": _fmt(report.get("expected_move"), "pct")},
        {"label": "Edge margin k", "value": _fmt(report.get("k"), "ratio")},
        {"label": "Filter", "value": "fired" if fired else "sat out"},
    ]
    branches = []
    for key, name in (
        ("filtered", "Filtered (trade only on edge)"),
        ("unfiltered", "Unfiltered (always trade)"),
    ):
        b = report.get(key) or {}
        pnl = b.get("trade_pnl")
        tone = "neutral"
        if isinstance(pnl, (int, float)):
            tone = "up" if pnl > 0 else "down" if pnl < 0 else "neutral"
        branches.append(
            {
                "name": name,
                "final_equity": _fmt_money(b.get("final_equity")),
                "trade_pnl": _fmt_money(pnl, signed=True),
                "pnl_tone": tone,
            }
        )
    return {"stats": stats, "branches": branches}


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


def _chat_history(raw: Any) -> list[tuple[str, str]] | None:
    """Validate the client-replayed transcript. Returns None on a malformed
    payload (the route 400s); caps are applied here so run_chat sees clean input."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    history: list[tuple[str, str]] = []
    for item in raw[-_HISTORY_MAX_MESSAGES:]:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        if role not in ("user", "assistant"):
            return None
        history.append((role, str(item.get("text", ""))[:_HISTORY_MAX_CHARS]))
    return history


def _planner_settings(raw: Any) -> str | None:
    """Render the /planner sizing strip into one opening-message line.

    Returns ``None`` for anything malformed (missing keys, non-numbers,
    out-of-range values) so a bad payload degrades to the no-settings flow
    instead of 400ing the chat.
    """
    if not isinstance(raw, dict):
        return None
    try:
        account = float(raw["account_size"])
        risk = float(raw["risk_per_trade_pct"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(account) or account <= 0 or not 0 < risk <= 0.2:
        return None
    return (
        f"Planner sizing (set on the page): account size ${account:,.0f}; "
        f"risk per trade {risk * 100:g}% ({risk:g})."
    )


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

    @app.get("/scans", response_class=HTMLResponse)
    def scans_latest(request: Request) -> HTMLResponse:
        dates = _scans.list_scan_dates()
        scan = _scans.load_scan(dates[0]) if dates else None
        return _TEMPLATES.TemplateResponse(request, "scans.html", {"dates": dates, "scan": scan})

    @app.get("/scans/{scan_date}", response_class=HTMLResponse)
    def scans_by_date(request: Request, scan_date: str) -> HTMLResponse:
        scan = _scans.load_scan(scan_date)
        if scan is None:
            return HTMLResponse("<p>scan not found</p>", status_code=404)
        return _TEMPLATES.TemplateResponse(
            request, "scans.html", {"dates": _scans.list_scan_dates(), "scan": scan}
        )

    @app.get("/models", response_class=HTMLResponse)
    def models(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "models.html",
            {"strategies": list(STRATEGIES.values()), "specs": list_specs()},
        )

    @app.get("/planner", response_class=HTMLResponse)
    def planner(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "planner.html", {})

    @app.get("/sentiment", response_class=HTMLResponse)
    def sentiment_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "sentiment.html", {})

    @app.get("/api/v1/sentiment/{ticker}")
    def sentiment_api(ticker: str, refresh: bool = False) -> JSONResponse:
        if not _sentiment.valid_ticker(ticker):
            return JSONResponse({"error": "invalid ticker"}, status_code=400)
        try:
            return JSONResponse(_sentiment.get_report(ticker, refresh=refresh))
        except Exception:  # engine degrades internally; reaching here is unexpected
            logger.exception("sentiment lookup failed for %s", ticker)
            return JSONResponse({"error": "sentiment lookup failed"}, status_code=500)

    @app.get("/tournaments", response_class=HTMLResponse)
    def tournaments_index(request: Request) -> HTMLResponse:
        ledger = _tournaments.load_ledger()
        return _TEMPLATES.TemplateResponse(
            request,
            "tournaments.html",
            {
                "stats": (ledger or {}).get("stats"),
                "rows": _tournaments.ledger_rows(ledger),
                "catalog": _tournaments.catalog(_scans.list_scan_dates()),
            },
        )

    @app.get("/tournaments/{scan_date}", response_class=HTMLResponse)
    def tournaments_by_date(request: Request, scan_date: str) -> HTMLResponse:
        scan = _scans.load_scan(scan_date)
        if scan is None:
            return HTMLResponse("<p>scan not found</p>", status_code=404)
        return _TEMPLATES.TemplateResponse(
            request,
            "tournament_day.html",
            {
                "dates": _scans.list_scan_dates(),
                "day": _tournaments.day_view(scan, _tournaments.load_ledger()),
            },
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
                "note": br.extra.get("note"),
                "report": _report_view(br.extra.get("report")),
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
        settings = _planner_settings(payload.get("settings"))
        history = _chat_history(payload.get("history"))
        if history is None:
            return JSONResponse(
                {"error": "history must be a list of {role: user|assistant, text} objects"},
                status_code=400,
            )
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
                for event in run_chat(
                    message,
                    provider,
                    Budget(),
                    context=context,
                    history=history or None,
                    settings=settings,
                ):
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
