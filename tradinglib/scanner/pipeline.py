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
from tradinglib.loaders.fundamentals.edgar import get_quarterly_trends
from tradinglib.loaders.fundamentals.yfinance import get_fundamental_snapshot
from tradinglib.loaders.universe.russell1000 import get_russell1000_constituents
from tradinglib.loaders.universe.sp500 import get_sp500_constituents
from tradinglib.scanner.briefs import brief_candidates
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.fa_gate import apply_edgar_trends, score_fundamentals
from tradinglib.scanner.rank import rank_candidates
from tradinglib.scanner.setups import detect_all
from tradinglib.tournament.run import run_tournament

logger = logging.getLogger(__name__)

_BENCHMARK = "SPY"


def _now() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def _gate_rows(scored: pd.DataFrame, gate_col: str, rank_col: str) -> list[dict]:
    """Report-ready rows for one FA cohort, ordered by rank."""
    cols = scored.loc[scored[gate_col], ["ticker", "name", "sector", "fa_score", rank_col]]
    return [
        {
            "ticker": rec["ticker"],
            "name": rec["name"],
            "sector": rec["sector"],
            "fa_score": float(rec["fa_score"]),
            "rank": int(rec[rank_col]),
        }
        for rec in cols.sort_values(rank_col).to_dict("records")
    ]


def _previous_winners(previous_report: dict | None) -> dict[tuple[str, str], str | None]:
    """(ticker, stance) -> last night's winner key (None = had no survivor)."""
    if not previous_report:
        return {}
    out: dict[tuple[str, str], str | None] = {}
    for stance, entries in (previous_report.get("tournament") or {}).items():
        for entry in entries:
            winner = entry.get("winner")
            out[(entry["ticker"], stance)] = winner["strategy"] if winner else None
    return out


def run_scan(
    config: ScanConfig,
    provider: LLMProvider | None = None,
    previous_report: dict | None = None,
) -> dict:
    """Run the full funnel and return the report-ready result dict."""
    asof = _now()
    start = (asof - pd.Timedelta(days=config.lookback_days)).strftime("%Y-%m-%d")

    if config.universe == "russell1000":
        universe = get_russell1000_constituents(refresh=config.refresh)
    elif config.universe == "sp500":
        universe = get_sp500_constituents(refresh=config.refresh)
    else:
        raise ValueError(f"unknown universe {config.universe!r}")
    universe_snapshot = universe.attrs.get("snapshot")
    if config.limit is not None:
        universe = universe.head(config.limit)

    errors: list[dict] = []
    if universe_snapshot and universe_snapshot != asof.strftime("%Y-%m-%d"):
        errors.append(
            {
                "ticker": "*",
                "stage": "universe",
                "error": f"stale universe snapshot {universe_snapshot} (scrape failed)",
            }
        )
    ciks = dict(zip(universe["ticker"], universe["cik"], strict=True))

    fundamentals = get_fundamental_snapshot(universe["ticker"].tolist(), refresh=config.refresh)
    # pass 1 keeps wider slates when the EDGAR pass will narrow them back down
    prelim_keep = config.fa_keep * 2 if config.edgar_enrich else config.fa_keep
    prelim_short = config.short_keep * 2 if config.edgar_enrich else config.short_keep
    scored = score_fundamentals(universe, fundamentals, keep=prelim_keep, short_keep=prelim_short)

    if config.edgar_enrich:
        rows: list[dict] = []
        in_gate = scored["passed_gate"] | scored["passed_short_gate"]
        for ticker in scored.loc[in_gate, "ticker"]:
            cik = ciks.get(ticker)
            if cik is None or pd.isna(cik):
                errors.append({"ticker": ticker, "stage": "edgar", "error": "no CIK mapping"})
                continue
            try:
                rows.append(
                    {
                        "ticker": ticker,
                        **get_quarterly_trends(int(cik), refresh=config.refresh),
                    }
                )
            except Exception as exc:
                errors.append({"ticker": ticker, "stage": "edgar", "error": str(exc)})
        trends = pd.DataFrame(
            rows,
            columns=[
                "ticker",
                "revenue_yoy",
                "revenue_yoy_prev",
                "revenue_accel",
                "eps_change_yoy",
            ],
        )
        scored = apply_edgar_trends(
            scored, trends, keep=config.fa_keep, short_keep=config.short_keep
        )

    shortlist = scored[scored["passed_gate"]]
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

    bars_failed: set[str] = {e["ticker"] for e in errors if e["stage"] == "bars"}

    if provider is not None and not config.skip_llm and candidates:
        brief_candidates(provider, candidates, ciks=ciks, errors=errors, refresh=config.refresh)

    fa_candidates = {
        "long": _gate_rows(scored, "passed_gate", "fa_rank"),
        "short": _gate_rows(scored, "passed_short_gate", "short_rank"),
    }

    tournament: dict[str, list[dict]] = {"long": [], "short": []}
    if not config.skip_strategies:
        prev_winners = _previous_winners(previous_report)
        t_start = (asof - pd.Timedelta(days=config.tournament_lookback_days)).strftime("%Y-%m-%d")
        for stance in ("long", "short"):
            for cand in fa_candidates[stance]:
                ticker = cand["ticker"]
                if ticker in bars_failed:
                    continue
                try:
                    # yfinance can append a price-less row (volume only) for the
                    # latest session; a NaN bar poisons ATR-based winner levels
                    t_bars = load_daily(ticker, start=t_start)
                    t_bars = t_bars.dropna(subset=["open", "high", "low", "close"])
                    tr = run_tournament(t_bars, stance)
                except Exception as exc:
                    errors.append({"ticker": ticker, "stage": "tournament", "error": str(exc)})
                    continue
                winner_key = tr.winner.key if tr.winner else None
                winner = None
                if tr.winner is not None and tr.winner_levels is not None:
                    winner = {**tr.winner.as_dict(), "levels": tr.winner_levels.as_dict()}
                tournament[stance].append(
                    {
                        "ticker": ticker,
                        "stance": stance,
                        "winner": winner,
                        "winner_changed": (
                            None
                            if (ticker, stance) not in prev_winners
                            else prev_winners[(ticker, stance)] != winner_key
                        ),
                        "survivors": [v.key for v in tr.verdicts if v.survived],
                        "n_trials": tr.n_trials,
                        "verdicts": [v.as_dict() for v in tr.verdicts],
                    }
                )

    return {
        "asof": asof.strftime("%Y-%m-%d"),
        "config": {
            "fa_keep": config.fa_keep,
            "short_keep": config.short_keep,
            "top": config.top,
            "limit": config.limit,
            "skip_llm": config.skip_llm,
            "edgar_enrich": config.edgar_enrich,
            "lookback_days": config.lookback_days,
            "earnings_warn_days": config.earnings_warn_days,
            "universe": config.universe,
            "skip_strategies": config.skip_strategies,
            "tournament_lookback_days": config.tournament_lookback_days,
        },
        "funnel": {
            "universe": len(universe),
            "fa_shortlist": len(shortlist),
            "fa_shortlist_short": int(scored["passed_short_gate"].sum()),
            "with_setups": len(candidates),
            "tournament_candidates": (
                0
                if config.skip_strategies
                else len(fa_candidates["long"]) + len(fa_candidates["short"])
            ),
            "tournament_survivors": sum(
                1 for entries in tournament.values() for e in entries if e["survivors"]
            ),
        },
        "fa_candidates": fa_candidates,
        "tournament": tournament,
        "candidates": rank_candidates(candidates, top=config.top),
        "errors": errors,
    }
