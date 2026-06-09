"""Daily forward chain snapshot for the earnings-straddle watchlist (Phase 2).

Run once per trading day (Windows Task Scheduler / cron — see
docs/data-sources.md). Idempotent: a ticker already snapshotted today is
skipped. Value accrues with calendar time; this forward dataset is the only
true point-in-time OOS source for the earnings straddle.

    uv run python scripts/collect_chain_snapshots.py
"""

from __future__ import annotations

from tradinglib.loaders.options.yf_chain import snapshot_chains

WATCHLIST = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "NFLX", "AMD"]


def main() -> None:
    for ticker, n in snapshot_chains(WATCHLIST).items():
        status = "already snapshotted today" if n < 0 else f"{n} rows"
        print(f"{ticker}: {status}")


if __name__ == "__main__":
    main()
