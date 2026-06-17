"""CLI for the forward ticket-performance ledger.

Examples:
    uv run python scripts/evaluate_tickets.py                    # rebuild the real ledger
    uv run python scripts/evaluate_tickets.py --base data/tmp/scans

Rebuilds ``ledger.json`` from every persisted scan report (the nightly Modal
cron does the same after each scan) and prints the cumulative stats plus one
line per ticket.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from tradinglib.data.paths import processed_dir
from tradinglib.loaders.equities.yfinance import corp_actions_since
from tradinglib.scanner.ledger import build_ledger, write_ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the ticket-performance ledger.")
    parser.add_argument("--base", type=Path, default=None, help="scan reports directory")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="re-score even closed (target/stopped) tickets instead of carrying the frozen record forward",
    )
    args = parser.parse_args(argv)

    base = args.base if args.base is not None else processed_dir("scans")
    ledger = build_ledger(
        base,
        asof=datetime.now(UTC).strftime("%Y-%m-%d"),
        actions_probe=corp_actions_since,
        force_rebuild=args.force_rebuild,
    )
    path = write_ledger(ledger, base)

    s = ledger["stats"]
    print(
        f"ledger {ledger['built_asof']}: {s['issued']} tickets — "
        f"{s['open']} open, {s['target']} target, {s['stopped']} stopped, "
        f"{s['waiting']} waiting, {s['expired']} expired, {s['errors']} errors"
    )
    for r in ledger["tickets"]:
        rr = f"{r['r']:+.2f}R" if isinstance(r.get("r"), (int, float)) else "-"
        print(
            f"  {r['date']} {r['ticker']:<6} {r['stance']:<5} "
            f"{r['strategy']:<12} {r['status']:<8} {rr}"
        )
    print(f"ledger: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
