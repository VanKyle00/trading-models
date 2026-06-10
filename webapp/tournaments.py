"""View models for the /tournaments pages: track record, catalog, day story.

Reads the per-date reports (via webapp.scans) and the forward ledger written
by the nightly job (tradinglib.scanner.ledger). Pure formatting and assembly —
no market-data access in the request path. Ledger records are heterogeneous
(error records lack outcome keys), so everything template-bound passes through
``_normalized`` first; templates must never see a missing key.
"""

from __future__ import annotations

from tradinglib.data.paths import processed_dir
from tradinglib.scanner import ledger as _ledger
from webapp.scans import SOURCE, load_scan

_SPARK_W, _SPARK_H, _SPARK_PAD = 160, 40, 2


def load_ledger() -> dict | None:
    """The forward ledger for the scans source, or None when absent/corrupt."""
    return _ledger.load_ledger(processed_dir(SOURCE))


def _normalized(rec: dict) -> dict:
    """A ledger record with every template-bound key present."""
    return {
        "levels": {},
        "r": None,
        "pct_move": None,
        "sessions_held": 0,
        "ambiguous_bar": False,
        "closes": [],
        "error": None,
        **rec,
    }


def sparkline_svg(record: dict) -> str:
    """Inline close-price sparkline with entry/stop/target rules; '' if too little data."""
    closes = record.get("closes") or []
    if len(closes) < 2:
        return ""
    levels = record.get("levels") or {}
    rules = [(k, levels.get(k)) for k in ("entry", "stop", "target")]
    values = [float(c) for _, c in closes]
    bounds = values + [float(v) for _, v in rules if isinstance(v, (int, float))]
    lo, hi = min(bounds), max(bounds)
    if hi <= lo:
        return ""

    def x(i: int) -> float:
        return round(i * _SPARK_W / (len(values) - 1), 1)

    def y(v: float) -> float:
        usable = _SPARK_H - 2 * _SPARK_PAD
        return round(_SPARK_H - _SPARK_PAD - (v - lo) * usable / (hi - lo), 1)

    parts = [
        f'<line class="sl-{k}" x1="0" y1="{y(float(v))}" x2="{_SPARK_W}" y2="{y(float(v))}"/>'
        for k, v in rules
        if isinstance(v, (int, float))
    ]
    points = " ".join(f"{x(i)},{y(v)}" for i, v in enumerate(values))
    parts.append(f'<polyline class="sl-price" points="{points}" fill="none"/>')
    return (
        f'<svg class="spark" viewBox="0 0 {_SPARK_W} {_SPARK_H}" width="{_SPARK_W}" '
        f'height="{_SPARK_H}" preserveAspectRatio="none" role="img" '
        f'aria-label="price since issue">{"".join(parts)}</svg>'
    )


def ledger_rows(ledger: dict | None) -> list[dict]:
    """Ledger records with rendered sparklines, newest first (build order)."""
    if not ledger:
        return []
    out = []
    for rec in ledger.get("tickets", []):
        norm = _normalized(rec)
        out.append({**norm, "spark": sparkline_svg(norm)})
    return out


def catalog(dates: list[str]) -> list[dict]:
    """One funnel-summary row per scan date, in the given date order."""
    rows = []
    for date in dates:
        scan = load_scan(date)
        if scan is None:
            continue
        funnel = scan.get("funnel", {})
        rows.append(
            {
                "date": date,
                "universe": funnel.get("universe"),
                "fa_long": funnel.get("fa_shortlist"),
                "fa_short": funnel.get("fa_shortlist_short"),
                "candidates": funnel.get("tournament_candidates"),
                "survivors": funnel.get("tournament_survivors"),
                "tickets": funnel.get("tickets"),
            }
        )
    return rows


def day_view(scan: dict, ledger: dict | None) -> dict:
    """The night's pipeline story: funnel + FA slates + tournament + tickets w/ outcomes."""
    date = scan.get("asof")
    outcomes = {
        (r.get("date"), r.get("ticker"), r.get("stance")): r
        for r in (ledger or {}).get("tickets", [])
    }
    tournament = scan.get("tournament") or {}
    entries = [e for stance in ("long", "short") for e in tournament.get(stance) or []]

    tickets: dict[str, list[dict]] = {}
    for stance in ("long", "short"):
        rows = []
        for t in (scan.get("tickets") or {}).get(stance) or []:
            rec = outcomes.get((date, t["ticker"], stance))
            outcome = None
            if rec is not None:
                norm = _normalized(rec)
                outcome = {**norm, "spark": sparkline_svg(norm)}
            rows.append({**t, "outcome": outcome})
        tickets[stance] = rows

    return {
        "scan": scan,
        "funnel": scan.get("funnel", {}),
        "fa_candidates": scan.get("fa_candidates"),
        "has_tournament": bool(entries),
        "tournament": entries,
        "n_survivor_tickers": sum(1 for e in entries if e.get("winner")),
        "tickets": tickets,
        "has_tickets": bool(tickets["long"] or tickets["short"]),
    }
