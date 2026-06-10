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
    candidates = result.get("candidates", [])
    if candidates:
        lines.extend(_md_table(candidates))
        lines.append("")
        lines.extend(_md_sections(candidates))
    else:
        lines.append("No setups forming today.")
        lines.append("")

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
    chronological. Returns ``None`` when there is no prior report (first run) —
    the tournament then marks ``winner_changed`` as unknown rather than false.
    """
    if not base.exists():
        return None
    dated = sorted(
        d for d in base.iterdir() if d.is_dir() and d.name < before and (d / "report.json").exists()
    )
    if not dated:
        return None
    return json.loads((dated[-1] / "report.json").read_text(encoding="utf-8"))
