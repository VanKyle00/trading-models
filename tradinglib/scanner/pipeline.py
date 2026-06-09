"""The scan orchestrator: universe → FA gate → setups → (briefs) → ranked report.

``run_scan`` is a pure function over the cached loaders so a CLI, a Modal
scheduled job, or the FastAPI workbench can all wrap it. Per-ticker work is
isolated: a ticker that fails any stage lands in ``result["errors"]`` and the
scan keeps going.

The LLM brief stage (Stage 3) runs only when a provider is injected AND
``config.skip_llm`` is false; until then candidates rank on FA + setup alone.
"""

from __future__ import annotations

import logging

import pandas as pd

from tradinglib.assistant.provider import LLMProvider
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.loaders.fundamentals.yfinance import get_fundamental_snapshot
from tradinglib.loaders.universe.sp500 import get_sp500_constituents
from tradinglib.scanner.briefs import brief_candidates
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.fa_gate import score_fundamentals
from tradinglib.scanner.rank import rank_candidates
from tradinglib.scanner.setups import detect_all

logger = logging.getLogger(__name__)

_BENCHMARK = "SPY"


def _now() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def run_scan(config: ScanConfig, provider: LLMProvider | None = None) -> dict:
    """Run the full funnel and return the report-ready result dict."""
    asof = _now()
    start = (asof - pd.Timedelta(days=config.lookback_days)).strftime("%Y-%m-%d")

    universe = get_sp500_constituents(refresh=config.refresh)
    if config.limit is not None:
        universe = universe.head(config.limit)

    fundamentals = get_fundamental_snapshot(universe["ticker"].tolist(), refresh=config.refresh)
    scored = score_fundamentals(universe, fundamentals, keep=config.fa_keep)
    shortlist = scored[scored["passed_gate"]]

    errors: list[dict] = []
    try:
        benchmark_close = load_daily(_BENCHMARK, start=start)["close"]
    except Exception as exc:  # benchmark is optional context, not a hard dependency
        logger.warning("benchmark load failed: %s", exc)
        benchmark_close = None
        errors.append({"ticker": _BENCHMARK, "stage": "benchmark", "error": str(exc)})

    candidates: list[dict] = []
    for row in shortlist.itertuples():
        ticker = row.ticker
        try:
            bars = load_daily(ticker, start=start)
            try:
                earnings = get_earnings_dates([ticker], refresh=config.refresh)
                earnings_dts = pd.DatetimeIndex(earnings["earnings_datetime"])
            except Exception as exc:
                errors.append({"ticker": ticker, "stage": "earnings", "error": str(exc)})
                earnings_dts = pd.DatetimeIndex([], tz="UTC")

            setups = detect_all(
                bars, benchmark_close=benchmark_close, earnings_datetimes=earnings_dts
            )
            if not setups:
                continue

            upcoming = earnings_dts[
                (earnings_dts > asof)
                & (earnings_dts <= asof + pd.Timedelta(days=config.earnings_warn_days))
            ]
            candidates.append(
                {
                    "ticker": ticker,
                    "name": getattr(row, "name", ticker),
                    "sector": row.sector,
                    "fa_score": float(row.fa_score),
                    "setup_score": max(s.score for s in setups),
                    "setups": [
                        {
                            "setup_type": s.setup_type,
                            "score": s.score,
                            "asof": s.asof,
                            "trigger_level": s.trigger_level,
                            "stop_level": s.stop_level,
                            "evidence": s.evidence,
                        }
                        for s in setups
                    ],
                    "qualitative_score": None,
                    "stance": None,
                    "red_flags": [],
                    "brief": None,
                    "earnings_warning": bool(len(upcoming) > 0),
                }
            )
        except Exception as exc:
            errors.append({"ticker": ticker, "stage": "bars", "error": str(exc)})

    if provider is not None and not config.skip_llm and candidates:
        ciks = dict(zip(universe["ticker"], universe["cik"], strict=True))
        brief_candidates(provider, candidates, ciks=ciks, errors=errors, refresh=config.refresh)

    return {
        "asof": asof.strftime("%Y-%m-%d"),
        "config": {
            "fa_keep": config.fa_keep,
            "top": config.top,
            "limit": config.limit,
            "skip_llm": config.skip_llm,
            "lookback_days": config.lookback_days,
            "earnings_warn_days": config.earnings_warn_days,
        },
        "funnel": {
            "universe": len(universe),
            "fa_shortlist": len(shortlist),
            "with_setups": len(candidates),
        },
        "candidates": rank_candidates(candidates, top=config.top),
        "errors": errors,
    }
