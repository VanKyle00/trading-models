"""CLI for the S&P 500 swing-setup scanner.

Examples:
    uv run python scripts/swing_scan.py --limit 10 --skip-llm   # smoke run
    uv run python scripts/swing_scan.py                          # full scan

Writes ``report.json`` + ``report.md`` under
``data/processed/scans/<YYYY-MM-DD>/`` (override with ``--out-dir``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tradinglib.data.paths import processed_dir
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.pipeline import run_scan
from tradinglib.scanner.report import write_report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the S&P 500 for swing setups.")
    parser.add_argument("--limit", type=int, default=None, help="truncate the universe (smoke)")
    parser.add_argument("--fa-keep", type=int, default=40, help="shortlist size after the FA gate")
    parser.add_argument("--top", type=int, default=15, help="max names in the ranked watchlist")
    parser.add_argument("--refresh", action="store_true", help="bust today's data caches")
    parser.add_argument("--skip-llm", action="store_true", help="stop after setup detection")
    parser.add_argument("--out-dir", type=Path, default=None, help="report directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ScanConfig(
        fa_keep=args.fa_keep,
        top=args.top,
        limit=args.limit,
        refresh=args.refresh,
        skip_llm=args.skip_llm,
    )

    result = run_scan(config, None)

    base = args.out_dir if args.out_dir is not None else processed_dir("scans")
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
    print(f"report: {json_path}")
    print(f"        {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
