"""Stage 4b: write the scan result as report.json + report.md.

The JSON file is the full machine-readable result; the Markdown file is the
human watchlist — funnel counts, a ranked table, then a per-ticker section
with setup evidence and (when present) the LLM brief.
"""

from __future__ import annotations

import json
from pathlib import Path


def _md_table(candidates: list[dict]) -> list[str]:
    lines = [
        "| # | Ticker | Setup | Score | Trigger | Stop | Flags |",
        "|---|--------|-------|-------|---------|------|-------|",
    ]
    for i, c in enumerate(candidates, start=1):
        setups = ", ".join(s["setup_type"] for s in c.get("setups", []))
        best = max(c.get("setups", []), key=lambda s: s["score"], default=None)
        trigger = f"{best['trigger_level']:.2f}" if best else "-"
        stop = f"{best['stop_level']:.2f}" if best else "-"
        flags = []
        if c.get("earnings_warning"):
            flags.append("earnings soon")
        if c.get("pinned"):
            flags.append(c.get("pinned_reason", "pinned"))
        lines.append(
            f"| {i} | {c['ticker']} | {setups} | {c['final_score']:.2f} "
            f"| {trigger} | {stop} | {', '.join(flags)} |"
        )
    return lines


def _md_sections(candidates: list[dict]) -> list[str]:
    lines: list[str] = []
    for c in candidates:
        lines.append(f"## {c['ticker']} — {c.get('name', '')}")
        lines.append("")
        lines.append(
            f"Sector: {c.get('sector', '?')} · FA score {c.get('fa_score', 0):.2f} "
            f"· setup score {c.get('setup_score', 0):.2f} · final {c['final_score']:.2f}"
        )
        if c.get("earnings_warning"):
            lines.append("")
            lines.append("⚠ Earnings within the warning window — event risk on the entry.")
        if c.get("pinned"):
            lines.append("")
            lines.append(f"⚠ Pinned to bottom — {c.get('pinned_reason')}")
        for s in c.get("setups", []):
            lines.append("")
            lines.append(
                f"**{s['setup_type']}** (score {s['score']:.2f}) — "
                f"trigger {s['trigger_level']:.2f}, stop {s['stop_level']:.2f}"
            )
            evidence = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in s.get("evidence", {}).items()
            )
            if evidence:
                lines.append(f"- evidence: {evidence}")
        brief = c.get("brief")
        if brief:
            lines.append("")
            lines.append(f"**LLM brief** ({brief.get('stance', '?')}): {brief.get('thesis', '')}")
            for label, key in (("Catalysts", "catalysts"), ("Risks", "risks")):
                items = brief.get(key) or []
                if items:
                    lines.append(f"- {label}: " + "; ".join(str(x) for x in items))
        lines.append("")
    return lines


_TICKET_NOTE = (
    "_Quotes are indicative last/close marks — re-check at the open, use limit "
    "orders. PoP is the market-implied probability from the chain IV, not an "
    "edge claim. Option risk figures are per share; one contract = 100 shares._"
)
_BORROW_FOOTNOTE = (
    "_Stock-short borrow/locate cost is not modeled — one reason short tickets "
    "prefer defined-risk option structures._"
)


def _fmt(value: object, spec: str = "{:.2f}", none: str = "-") -> str:
    return spec.format(value) if isinstance(value, (int, float)) else none


def _size(structure: dict) -> str:
    qty = structure.get("quantity")
    if not qty:
        return "UNSIZED"
    return f"{qty} {structure.get('unit', 'unit')}{'s' if qty != 1 else ''}"


def _ticket_flags(ticket: dict) -> str:
    """Terse per-ticket flags for the summary table (the indicative warning is
    section-level, not a flag; full warnings appear in the detail block)."""
    flags = [w.split(":")[0] for w in ticket.get("warnings", []) if not w.startswith("indicative")]
    if ticket.get("evidence", {}).get("winner_changed"):
        flags.insert(0, "winner changed")
    return ", ".join(flags)


def _ticket_table(tickets: list[dict]) -> list[str]:
    lines = [
        "| Ticker | Winner | Entry | Stop | Target | Recommended | Size | Flags |",
        "|--------|--------|-------|------|--------|-------------|------|-------|",
    ]
    for t in tickets:
        lv = t["levels"]
        rec = next((s for s in t["structures"] if s.get("recommended")), t["structures"][0])
        lines.append(
            f"| {t['ticker']} | {t['strategy']} | {lv['entry']:.2f} | {lv['stop']:.2f} "
            f"| {lv['target']:.2f} | {rec['label']} | {_size(rec)} | {_ticket_flags(t)} |"
        )
    return lines


def _ticket_sections(tickets: list[dict]) -> list[str]:
    lines: list[str] = []
    for t in tickets:
        lv, ev = t["levels"], t.get("evidence", {})
        lines.append(f"#### {t['ticker']} — {t['strategy']} ({t['stance']})")
        lines.append("")
        lines.append(
            f"Entry {lv['entry']:.2f} ({lv['entry_type']}) · stop {lv['stop']:.2f} "
            f"· target {lv['target']:.2f} — {lv['condition']}"
        )
        lines.append("")
        evidence = (
            f"OOS evidence: DSR {_fmt(ev.get('deflated_sharpe'), '{:.3f}')} · "
            f"Sharpe {_fmt(ev.get('sharpe'))} · {ev.get('n_trades', '?')} trades "
            f"over {ev.get('n_windows', '?')} windows"
        )
        if ev.get("winner_changed"):
            evidence += " · **winner changed overnight**"
        lines.append(evidence)
        meta = []
        if t.get("quotes_asof"):
            meta.append(f"quotes as of {t['quotes_asof']}")
        if isinstance(t.get("iv_ratio"), (int, float)):
            meta.append(f"IV/RV {t['iv_ratio']:.2f}")
        if t.get("next_earnings"):
            meta.append(f"next earnings {t['next_earnings']}")
        if meta:
            lines.append("")
            lines.append(" · ".join(meta))
        lines.append("")
        lines.append("| Structure | Size | Max loss | Max gain | Breakeven | PoP | R:R |")
        lines.append("|-----------|------|----------|----------|-----------|-----|-----|")
        for s in t["structures"]:
            label = f"**{s['label']}**" if s.get("recommended") else s["label"]
            lines.append(
                f"| {label} | {_size(s)} | {_fmt(s.get('max_loss'))} "
                f"| {_fmt(s.get('max_gain'), none='uncapped')} "
                f"| {_fmt(s.get('breakeven'))} "
                f"| {_fmt(s.get('pop_market_implied'), '{:.0%}')} "
                f"| {_fmt(s.get('rr'))} |"
            )
        lines.append("")
        for w in t.get("warnings", []):
            lines.append(f"- ⚠ {w}")
        for s in t["structures"]:
            for w in s.get("warnings", []):
                lines.append(f"- ⚠ {s['label']}: {w}")
        lines.append("")
    return lines


def render_markdown(result: dict) -> str:
    funnel = result.get("funnel", {})
    lines = [
        f"# Swing scan — {result.get('asof', '?')}",
        "",
        f"Funnel: {funnel.get('universe', '?')} universe → "
        f"{funnel.get('fa_shortlist', '?')} past FA gate → "
        f"{funnel.get('with_setups', '?')} with setups forming",
        "",
    ]
    if "tournament_candidates" in funnel:
        lines += [
            f"Tournament: {funnel.get('tournament_candidates', '?')} candidates → "
            f"{funnel.get('tournament_survivors', '?')} with surviving strategies → "
            f"{funnel.get('tickets', '?')} tickets",
            "",
        ]
    candidates = result.get("candidates", [])
    if candidates:
        lines.extend(_md_table(candidates))
        lines.append("")
        lines.extend(_md_sections(candidates))
    else:
        lines.append("No setups forming today.")
        lines.append("")

    tickets = result.get("tickets") or {}
    if any(tickets.get(stance) for stance in ("long", "short")):
        lines.append("## Tickets")
        lines.append("")
        lines.append(_TICKET_NOTE)
        lines.append("")
        for stance, title in (("long", "Long"), ("short", "Short")):
            entries = tickets.get(stance) or []
            if not entries:
                continue
            lines.append(f"### {title}")
            lines.append("")
            lines.extend(_ticket_table(entries))
            lines.append("")
            if stance == "short":
                lines.append(_BORROW_FOOTNOTE)
                lines.append("")
            lines.extend(_ticket_sections(entries))

    errors = result.get("errors", [])
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- {e.get('ticker', '?')} ({e.get('stage', '?')}): {e.get('error', '')}")
        lines.append("")
    return "\n".join(lines)


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def load_latest_report(base: Path, *, before: str) -> dict | None:
    """Latest ``report.json`` under ``base`` from a date strictly before ``before``.

    Scan directories are named ``YYYY-MM-DD`` so lexicographic order is
    chronological. Returns ``None`` when there is no prior report (first run)
    or the latest one is unreadable/corrupt (e.g. a partial write) — the
    previous report is optional context and must never kill tonight's scan;
    the tournament then marks ``winner_changed`` as unknown rather than false.
    """
    if not base.exists():
        return None
    dated = sorted(
        d for d in base.iterdir() if d.is_dir() and d.name < before and (d / "report.json").exists()
    )
    if not dated:
        return None
    try:
        return json.loads((dated[-1] / "report.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
