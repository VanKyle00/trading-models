"""Forward ticket-performance ledger: every nightly ticket, scored from daily bars.

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
from tradinglib.strategist.evaluate import simulate_ticket

LEDGER_FILENAME = "ledger.json"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Loader = Callable[..., pd.DataFrame]


def _issues(base: Path) -> list[tuple[str, dict]]:
    """(issue_date, ticket) for every ticket in every report, oldest date first."""
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
    }


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
            record.update(simulate_ticket(ticket, bars, asof=date))
        except Exception as exc:
            record.update(status="error", error=str(exc))
        records.append(record)
    records.sort(
        key=lambda r: r["date"], reverse=True
    )  # newest issue date first, stable within date
    return {"built_asof": asof, "stats": _stats(records), "tickets": records}


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
