"""CLI for the swing-setup scanner.

Examples:
    uv run python scripts/swing_scan.py --limit 10 --skip-llm   # smoke run
    uv run python scripts/swing_scan.py                          # full scan

Writes ``report.json`` + ``report.md`` under
``data/processed/scans/<YYYY-MM-DD>/`` (override with ``--out-dir``).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from tradinglib.data.paths import processed_dir
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.pipeline import run_scan
from tradinglib.scanner.report import load_latest_report, write_report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the constituent universe for swing setups.")
    parser.add_argument("--limit", type=int, default=None, help="truncate the universe (smoke)")
    parser.add_argument("--fa-keep", type=int, default=40, help="shortlist size after the FA gate")
    parser.add_argument("--top", type=int, default=15, help="max names in the ranked watchlist")
    parser.add_argument("--refresh", action="store_true", help="bust today's data caches")
    parser.add_argument("--skip-llm", action="store_true", help="stop after setup detection")
    parser.add_argument(
        "--skip-edgar", action="store_true", help="skip the EDGAR quarterly-trend gate pass"
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="report directory")
    parser.add_argument(
        "--universe",
        choices=("russell1000", "sp500"),
        default="russell1000",
        help="constituent universe to scan",
    )
    parser.add_argument(
        "--short-keep",
        type=int,
        default=40,
        help="short-candidate slate size (bottom of the FA ranking; 0 disables)",
    )
    parser.add_argument(
        "--skip-strategies",
        action="store_true",
        help="skip the per-ticker strategy tournament over the FA candidates",
    )
    parser.add_argument(
        "--account-size",
        type=float,
        default=100_000.0,
        help="account size the ticket playbook sizes risk against",
    )
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=0.01,
        help="per-ticket risk budget as a fraction of account size",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ScanConfig(
        fa_keep=args.fa_keep,
        short_keep=args.short_keep,
        top=args.top,
        limit=args.limit,
        refresh=args.refresh,
        skip_llm=args.skip_llm,
        edgar_enrich=not args.skip_edgar,
        universe=args.universe,
        skip_strategies=args.skip_strategies,
        account_size=args.account_size,
        risk_per_trade_pct=args.risk_per_trade_pct,
    )

    provider = None
    if not args.skip_llm:
        # lazy import: --skip-llm runs need neither the anthropic SDK nor a key
        from dotenv import load_dotenv

        from tradinglib.assistant.provider import ClaudeProvider

        load_dotenv()  # pick up ANTHROPIC_API_KEY from a local .env
        provider = ClaudeProvider()

    base = args.out_dir if args.out_dir is not None else processed_dir("scans")
    previous = load_latest_report(base, before=datetime.now(UTC).strftime("%Y-%m-%d"))
    result = run_scan(config, provider, previous_report=previous)

    json_path, md_path = write_report(result, base / result["asof"])

    funnel = result["funnel"]
    print(
        f"scan {result['asof']}: {funnel['universe']} universe -> "
        f"{funnel['fa_shortlist']} past FA gate -> {funnel['with_setups']} with setups"
    )
    for i, c in enumerate(result["candidates"], start=1):
        setups = ", ".join(s["setup_type"] for s in c["setups"])
        flag = " [pinned]" if c["pinned"] else ""
        print(f"{i:>3}. {c['ticker']:<6} {c['final_score']:.2f}  {setups}{flag}")
    if result["errors"]:
        print(f"{len(result['errors'])} ticker(s) errored — see the report for details")
    print(
        f"FA candidates: {len(result['fa_candidates']['long'])} long, "
        f"{len(result['fa_candidates']['short'])} short"
    )
    print(
        f"tournament: {result['funnel']['tournament_candidates']} candidates -> "
        f"{result['funnel']['tournament_survivors']} with surviving strategies"
    )
    for stance in ("long", "short"):
        for entry in result["tournament"][stance]:
            winner = entry["winner"]
            if winner:
                lv = winner["levels"]
                changed = " [winner changed]" if entry["winner_changed"] else ""
                print(
                    f"  {entry['ticker']:<6} {stance:<5} {winner['strategy']:<10} "
                    f"entry {lv['entry']:.2f} stop {lv['stop']:.2f} "
                    f"target {lv['target']:.2f}{changed}"
                )
    print(f"tickets: {result['funnel']['tickets']}")
    for stance in ("long", "short"):
        for t in result["tickets"][stance]:
            rec = next(s for s in t["structures"] if s["recommended"])
            qty = rec["quantity"] if rec["quantity"] else "UNSIZED"
            print(f"  {t['ticker']:<6} {stance:<5} {t['strategy']:<10} -> {rec['label']} x{qty}")
    print(f"report: {json_path}")
    print(f"        {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
