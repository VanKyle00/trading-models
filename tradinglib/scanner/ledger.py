"""Forward ticket-performance ledger: every nightly ticket, scored from daily bars.

Both tiers (ticket and watch) are tracked identically so the tiering itself is
empirically validated by forward performance; deliberately NO ledger Sharpe —
annualizing a sparse, irregular trade series is noise dressed as rigor (see
docs/methodology.md).

Rebuilt from scratch on every run (idempotent, self-healing, cheap at a
handful of tickets a night) and written next to the per-date reports as
``scans/ledger.json``. A bad ticker can never fail the build: per-ticket
failures become ``status: "error"`` records, and the nightly cron wraps the
whole build so a ledger failure can never lose the night's scan report.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.strategist.evaluate import ENTRY_WINDOW, simulate_ticket

LEDGER_FILENAME = "ledger.json"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Loader = Callable[..., pd.DataFrame]


def _issues(base: Path) -> list[tuple[str, dict]]:
    """(issue_date, ticket) for every ticket/watch-with-levels in every report, oldest date first."""
    if not base.exists():
        return []
    out: list[tuple[str, dict]] = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and _DATE_RE.match(p.name)):
        path = d / "report.json"
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # a partial/corrupt old report must not kill the ledger
        tickets = report.get("tickets") or {}
        for stance in ("long", "short"):
            for ticket in tickets.get(stance) or []:
                out.append((d.name, ticket))
        watchlist = report.get("watchlist") or {}
        for stance in ("long", "short"):
            for row in watchlist.get(stance) or []:
                if not row.get("levels"):
                    continue  # report-only rows: nothing to simulate
                out.append((d.name, row))
    return out


def _stats(records: list[dict]) -> dict:
    counts = Counter(r["status"] for r in records)
    wins, losses = counts.get("target", 0), counts.get("stopped", 0)
    closed = wins + losses
    rs = [r["r"] for r in records if isinstance(r.get("r"), (int, float))]
    return {
        "issued": len(records),
        "waiting": counts.get("waiting", 0),
        "expired": counts.get("expired", 0),
        "open": counts.get("open", 0),
        "stopped": losses,
        "target": wins,
        "errors": counts.get("error", 0),
        "hit_rate": wins / closed if closed else None,
        "total_r": float(sum(rs)),
        "avg_r": float(sum(rs) / len(rs)) if rs else None,
        "max_drawdown_r": _max_drawdown_r(records),
    }


def _max_drawdown_r(records: list[dict]) -> float | None:
    closed = [
        r
        for r in records
        if r.get("status") in ("target", "stopped") and isinstance(r.get("r"), (int, float))
    ]
    if not closed:
        return None
    closed.sort(key=lambda r: (r.get("exit_date") or r["date"], r["ticker"]))
    cum = peak = dd = 0.0
    for r in closed:
        cum += r["r"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return float(dd)


def build_ledger(base: Path, *, asof: str, loader: Loader = load_daily) -> dict:
    """Score every persisted ticket against daily bars. One loader call per ticker."""
    bars_by_ticker: dict[str, pd.DataFrame | Exception] = {}
    records: list[dict] = []
    for date, ticket in _issues(base):
        ticker = ticket["ticker"]
        record: dict = {
            "date": date,
            "ticker": ticker,
            "stance": ticket["stance"],
            "strategy": ticket["strategy"],
            "tier": ticket.get("tier", "ticket"),
        }
        if ticker not in bars_by_ticker:
            try:
                # oldest issue first, so `start` covers every later re-issue too;
                # refresh=True is load-bearing: the parquet cache on the Modal Volume
                # is written at issue night, so without a forced download the ledger
                # would see zero post-issue bars and every ticket would stay "waiting"
                # forever (one fresh download per distinct ledgered ticker per build)
                bars_by_ticker[ticker] = loader(ticker, start=date, refresh=True)
            except Exception as exc:  # cache the failure: don't re-hit a dead ticker
                bars_by_ticker[ticker] = exc
        bars = bars_by_ticker[ticker]
        try:
            if isinstance(bars, Exception):
                raise bars
            record["levels"] = {
                k: ticket["levels"][k] for k in ("entry", "entry_type", "stop", "target")
            }
            record["entry_window"] = int(ticket.get("entry_window", ENTRY_WINDOW))
            record.update(
                simulate_ticket(ticket, bars, asof=date, entry_window=record["entry_window"])
            )
        except Exception as exc:
            record.update(status="error", error=str(exc))
        records.append(record)
    records.sort(
        key=lambda r: r["date"], reverse=True
    )  # newest issue date first, stable within date
    stats = {
        **_stats(records),
        "by_tier": {t: _stats([r for r in records if r["tier"] == t]) for t in ("ticket", "watch")},
    }
    return {"built_asof": asof, "stats": stats, "tickets": records}


def write_ledger(ledger: dict, base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / LEDGER_FILENAME
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return path


def load_ledger(base: Path) -> dict | None:
    """The parsed ledger, or None if absent/corrupt (pages render without it)."""
    path = base / LEDGER_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
