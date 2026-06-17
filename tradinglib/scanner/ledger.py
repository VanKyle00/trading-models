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
_LEDGER_SCHEMA_VERSION = 1  # bump to invalidate prior frozen records on a format change
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CLOSED_STATUSES = ("target", "stopped")

Loader = Callable[..., pd.DataFrame]
# (ticker, issue_date, exit_date) -> True if a split/dividend ex-date falls in (issue, exit]
ActionsProbe = Callable[[str, str, str], bool]


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


def _realized(records: list[dict]) -> list[dict]:
    """Closed trades that count toward the yardstick: target/stopped with a numeric
    R, EXCLUDING any whose holding window crossed a split/dividend. ``auto_adjust``
    re-scales the refreshed bars, so a corp-action ticket's frozen-level score is
    unreliable; it is dropped rather than trusted (audit A5b)."""
    return [
        r
        for r in records
        if r.get("status") in _CLOSED_STATUSES
        and isinstance(r.get("r"), (int, float))
        and not r.get("corp_action_since_issue")
    ]


def _stats(records: list[dict]) -> dict:
    counts = Counter(r["status"] for r in records)
    # Realized R aggregates ONLY closed (and non-corp-action) trades. An open
    # position's r is a mark-to-last-close; folding it into total_r/avg_r would let
    # an unrealized mark inflate the headline used to compare models, so it is
    # reported separately as open_unrealized_r.
    realized_records = _realized(records)
    realized = [r["r"] for r in realized_records]
    wins = sum(1 for r in realized_records if r["status"] == "target")
    closed = len(realized_records)
    open_rs = [
        r["r"]
        for r in records
        if r.get("status") == "open" and isinstance(r.get("r"), (int, float))
    ]
    return {
        "issued": len(records),
        "waiting": counts.get("waiting", 0),
        "expired": counts.get("expired", 0),
        "open": counts.get("open", 0),
        "stopped": counts.get("stopped", 0),
        "target": counts.get("target", 0),
        "errors": counts.get("error", 0),
        "corp_action_excluded": sum(1 for r in records if r.get("corp_action_since_issue")),
        "hit_rate": wins / closed if closed else None,
        "total_r": float(sum(realized)),
        "avg_r": float(sum(realized) / len(realized)) if realized else None,
        "open_unrealized_r": float(sum(open_rs)) if open_rs else None,
        "max_drawdown_r": _max_drawdown_r(records),
    }


def _max_drawdown_r(records: list[dict]) -> float | None:
    closed = _realized(records)
    if not closed:
        return None
    closed.sort(key=lambda r: (r.get("exit_date") or r["date"], r["ticker"]))
    cum = peak = dd = 0.0
    for r in closed:
        cum += r["r"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return float(dd)


def _frozen_closed(base: Path, *, force_rebuild: bool) -> dict[tuple, dict]:
    """Prior-build closed (target/stopped) records, keyed for carry-forward.

    A closed ticket's status+R is frozen: once decided it must never change on a
    later build, even though ``refresh=True`` re-downloads ``auto_adjust`` bars
    that a split/dividend may have silently re-scaled (audit A5b). ``force_rebuild``
    discards the prior ledger so a bad early close can be corrected.
    """
    if force_rebuild:
        return {}
    prior = load_ledger(base)
    if not prior or prior.get("schema_version") != _LEDGER_SCHEMA_VERSION:
        return {}
    out: dict[tuple, dict] = {}
    for r in prior.get("tickets", []):
        if r.get("status") in _CLOSED_STATUSES:
            key = (r["date"], r["ticker"], r["stance"], r["strategy"], r.get("tier", "ticket"))
            out[key] = r
    return out


def build_ledger(
    base: Path,
    *,
    asof: str,
    loader: Loader = load_daily,
    actions_probe: ActionsProbe | None = None,
    force_rebuild: bool = False,
) -> dict:
    """Score every persisted ticket against daily bars. One loader call per ticker.

    Closed (target/stopped) tickets are frozen from the prior build and carried
    forward verbatim — their status/R never changes once decided (audit A5b); only
    waiting/open/error rows are re-simulated. ``force_rebuild`` re-scores
    everything. When ``actions_probe`` is supplied, a ticket whose holding window
    crossed a split/dividend is flagged ``corp_action_since_issue`` and excluded
    from realized stats (its frozen levels predate the ``auto_adjust`` rescale).
    """
    frozen = _frozen_closed(base, force_rebuild=force_rebuild)
    bars_by_ticker: dict[str, pd.DataFrame | Exception] = {}
    records: list[dict] = []
    for date, ticket in _issues(base):
        ticker = ticket["ticker"]
        key = (date, ticker, ticket["stance"], ticket["strategy"], ticket.get("tier", "ticket"))
        if key in frozen:
            records.append(frozen[key])  # closed once -> immutable; no re-fetch, no re-score
            continue
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
        if actions_probe is not None and record.get("status") in _CLOSED_STATUSES:
            # Flag a corp action in the holding window on the SAME build that first
            # closes the ticket, BEFORE it is frozen, so a mis-scaled close is never
            # cemented into the realized stats. Best-effort: never fail the build.
            try:
                if actions_probe(ticker, date, record["exit_date"]):
                    record["corp_action_since_issue"] = True
            except Exception:
                pass
        records.append(record)
    records.sort(
        key=lambda r: r["date"], reverse=True
    )  # newest issue date first, stable within date
    stats = {
        **_stats(records),
        "by_tier": {t: _stats([r for r in records if r["tier"] == t]) for t in ("ticket", "watch")},
    }
    return {
        "built_asof": asof,
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "stats": stats,
        "tickets": records,
    }


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
