"""No-lookahead historical replay of the technical scan funnel.

Examples:
    uv run python scripts/backfill_scan.py --smoke          # one night, timed
    uv run python scripts/backfill_scan.py                  # full replay
    uv run python scripts/backfill_scan.py --days 300 --step 5

Replays setups -> tournament -> FDR -> tiers as-of past nights using the same
pipeline functions the nightly cron runs, then forward-scores every issued
row with the same ``simulate_ticket`` the live ledger uses. Each night sees
bars only up to its own date (asserted), with the tournament/setup windows
sliding exactly like prod.

Honesty caveats, by construction:
- The FA gate is NOT replayable (yfinance fundamentals are current-only), so
  the family is frozen to the latest real report's 40 long + 40 short names.
  Selection-layer survivorship: today's fundamentally-healthy names likely
  flatter the long side. Everything downstream of selection is leak-free.
- Earnings dates come from today's fetch (limited history), so the oldest
  nights may see slightly thinner earnings flags than a live run would have.
- LLM briefs, option chains/structures and winner_changed are skipped: none
  of them gate ticket issuance, and the ledger scores stock-level levels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradinglib.data.paths import processed_dir
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.ledger import _stats
from tradinglib.scanner.setups import detect_all
from tradinglib.scanner.tiers import apply_fdr, build_watchlist
from tradinglib.strategist.evaluate import simulate_ticket
from tradinglib.tournament.run import run_tournament

_BENCHMARK = "SPY"


def _naive_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """Bars on a tz-naive UTC index with price-less rows dropped."""
    idx = frame.index
    if getattr(idx, "tz", None) is not None:
        frame = frame.set_axis(idx.tz_convert("UTC").tz_localize(None))
    return frame.dropna(subset=["open", "high", "low", "close"])


def _family(report_path: Path) -> dict[str, list[str]]:
    """The frozen FA family: {stance: [tickers]} from a real report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fa = report["fa_candidates"]
    return {stance: [row["ticker"] for row in fa[stance]] for stance in ("long", "short")}


def _slice(bars: pd.DataFrame, asof: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """Bars in (asof - window_days, asof] — the no-lookahead view of one night."""
    out = bars.loc[(bars.index >= asof - pd.Timedelta(days=window_days)) & (bars.index <= asof)]
    assert len(out) == 0 or out.index.max() <= asof, "lookahead leak in night slice"
    return out


def _earnings_flags(index: pd.DatetimeIndex, dts: pd.DatetimeIndex) -> pd.Series:
    """Prod's earnings-column construction over an already-sliced index."""
    flags = pd.Series(False, index=index)
    if len(dts):
        pos = index.searchsorted(dts)
        pos = pos[pos < len(index)]
        flags.iloc[pos] = True
    return flags


def run_night(
    asof: pd.Timestamp,
    family: dict[str, list[str]],
    bars_by_ticker: dict[str, pd.DataFrame],
    earnings_by_ticker: dict[str, pd.DatetimeIndex],
    config: ScanConfig,
) -> tuple[dict, list[dict], list[str]]:
    """One replayed night: (tournament-funnel summary, issued rows, errors)."""
    errors: list[str] = []
    tournament: dict[str, list[dict]] = {"long": [], "short": []}

    for stance in ("long", "short"):
        for ticker in family[stance]:
            bars = bars_by_ticker.get(ticker)
            if bars is None:
                continue
            t_bars = _slice(bars, asof, config.tournament_lookback_days)
            flags = _earnings_flags(
                t_bars.index, earnings_by_ticker.get(ticker, pd.DatetimeIndex([]))
            )
            try:
                tr = run_tournament(t_bars.assign(earnings=flags), stance)
            except Exception as exc:
                errors.append(f"{ticker} {stance}: {exc}")
                continue
            winner = None
            if tr.winner is not None and tr.winner_levels is not None:
                winner = {**tr.winner.as_dict(), "levels": tr.winner_levels.as_dict()}
            tournament[stance].append(
                {
                    "ticker": ticker,
                    "stance": stance,
                    "winner": winner,
                    "survivors": [v.key for v in tr.verdicts if v.survived],
                    "n_trials": tr.n_trials,
                    "verdicts": [v.as_dict() for v in tr.verdicts],
                }
            )

    benchmark = bars_by_ticker.get(_BENCHMARK)
    bench_close = (
        _slice(benchmark, asof, config.lookback_days)["close"] if benchmark is not None else None
    )
    candidates: list[dict] = []
    for cohort, tickers in [("long", family["long"]), ("short", family["short"])]:
        for ticker in tickers:
            bars = bars_by_ticker.get(ticker)
            if bars is None:
                continue
            s_bars = _slice(bars, asof, config.lookback_days)
            try:
                # detect_pead self-filters to events <= the sliced last bar
                setups = detect_all(
                    s_bars,
                    stance=cohort,
                    benchmark_close=bench_close,
                    earnings_datetimes=earnings_by_ticker.get(ticker, pd.DatetimeIndex([])),
                )
            except Exception as exc:
                errors.append(f"{ticker} setups: {exc}")
                continue
            if setups:
                candidates.append(
                    {
                        "ticker": ticker,
                        "cohort": cohort,
                        "setups": [
                            {
                                "setup_type": s.setup_type,
                                "score": s.score,
                                "trigger_level": s.trigger_level,
                                "stop_level": s.stop_level,
                            }
                            for s in setups
                        ],
                    }
                )

    fdr_passed, _, fdr_family = apply_fdr(tournament, config.fdr_alpha)
    watchlist = build_watchlist(
        tournament, candidates, fdr_passed, watch_dsr_floor=config.watch_dsr_floor
    )

    date = asof.strftime("%Y-%m-%d")
    issued: list[dict] = []
    for stance in ("long", "short"):
        for entry in tournament[stance]:
            if entry["winner"] is None or not fdr_passed.get((stance, entry["ticker"]), False):
                continue
            levels = entry["winner"]["levels"]
            issued.append(
                {
                    "date": date,
                    "ticker": entry["ticker"],
                    "stance": stance,
                    "strategy": entry["winner"]["strategy"],
                    "tier": "ticket",
                    "levels": {k: levels[k] for k in ("entry", "entry_type", "stop", "target")},
                }
            )
        for row in watchlist[stance]:
            if not row.get("levels"):
                continue  # report-only watch rows: nothing to simulate
            issued.append(
                {
                    "date": date,
                    **{k: row[k] for k in ("ticker", "stance", "strategy", "tier", "levels")},
                }
            )

    funnel = {
        "date": date,
        "tournaments": sum(len(v) for v in tournament.values()),
        "survivors": sum(1 for v in tournament.values() for e in v if e["survivors"]),
        "fdr_family": fdr_family,
        "fdr_passed": sum(1 for v in fdr_passed.values() if v),
        "tickets": sum(1 for r in issued if r["tier"] == "ticket"),
        "watch": sum(1 for r in issued if r["tier"] == "watch"),
        "best_dsr": max(
            (v["deflated_sharpe"] for t in tournament.values() for e in t for v in e["verdicts"]),
            default=None,
        ),
        "errors": len(errors),
    }
    return funnel, issued, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the technical scan funnel, no lookahead.")
    parser.add_argument("--days", type=int, default=300, help="calendar days to replay")
    parser.add_argument("--step", type=int, default=5, help="sessions between replay nights")
    parser.add_argument("--report", type=Path, default=None, help="report.json for the FA family")
    parser.add_argument("--out", type=Path, default=None, help="output JSON path")
    parser.add_argument("--smoke", action="store_true", help="run only the oldest night, timed")
    args = parser.parse_args(argv)

    config = ScanConfig()
    today = pd.Timestamp(datetime.now(UTC).strftime("%Y-%m-%d"))
    replay_start = today - pd.Timedelta(days=args.days)
    bar_start = (replay_start - pd.Timedelta(days=config.tournament_lookback_days)).strftime(
        "%Y-%m-%d"
    )

    report_path = args.report
    if report_path is None:
        dates = sorted(
            p.name for p in processed_dir("scans").iterdir() if (p / "report.json").exists()
        )
        report_path = processed_dir("scans") / dates[-1] / "report.json"
    family = _family(report_path)
    tickers = sorted(set(family["long"]) | set(family["short"]))
    print(f"family: {len(family['long'])} long / {len(family['short'])} short from {report_path}")

    bars_by_ticker: dict[str, pd.DataFrame] = {}
    earnings_by_ticker: dict[str, pd.DatetimeIndex] = {}
    for ticker in [*tickers, _BENCHMARK]:
        try:
            bars_by_ticker[ticker] = _naive_utc(load_daily(ticker, start=bar_start))
        except Exception as exc:
            print(f"  bars failed for {ticker}: {exc}")
            continue
        if ticker == _BENCHMARK:
            continue
        try:
            earnings = get_earnings_dates([ticker])
            dts = pd.DatetimeIndex(earnings["earnings_datetime"])
            if dts.tz is None:
                dts = dts.tz_localize("UTC")
            earnings_by_ticker[ticker] = dts.tz_convert("UTC").tz_localize(None)
        except Exception:
            earnings_by_ticker[ticker] = pd.DatetimeIndex([])
    print(f"bars loaded: {len(bars_by_ticker)}/{len(tickers) + 1}")

    calendar = bars_by_ticker[_BENCHMARK].loc[lambda b: b.index >= replay_start].index
    nights = list(calendar[:: args.step])
    if args.smoke:
        nights = nights[:1]
    print(f"replaying {len(nights)} nights ({nights[0].date()} .. {nights[-1].date()})")

    funnels: list[dict] = []
    records: list[dict] = []
    all_errors: list[str] = []
    for i, asof in enumerate(nights, 1):
        t0 = time.monotonic()
        funnel, issued, errors = run_night(asof, family, bars_by_ticker, earnings_by_ticker, config)
        for row in issued:
            record = dict(row)
            try:
                record.update(simulate_ticket(row, bars_by_ticker[row["ticker"]], asof=row["date"]))
            except Exception as exc:
                record.update(status="error", error=str(exc))
            records.append(record)
        funnels.append(funnel)
        all_errors.extend(errors)
        print(
            f"[{i}/{len(nights)}] {funnel['date']}: survivors={funnel['survivors']} "
            f"fdr_passed={funnel['fdr_passed']} tickets={funnel['tickets']} "
            f"watch={funnel['watch']} best_dsr={funnel['best_dsr'] or 0:.3f} "
            f"errors={funnel['errors']} ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )

    stats = {
        **_stats(records),
        "by_tier": {t: _stats([r for r in records if r["tier"] == t]) for t in ("ticket", "watch")},
        "by_stance": {
            s: _stats([r for r in records if r["stance"] == s]) for s in ("long", "short")
        },
    }
    out_path = args.out or (
        processed_dir("backfill")
        / f"replay_{nights[0].strftime('%Y%m%d')}_{nights[-1].strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_asof": today.strftime("%Y-%m-%d"),
                "family_report": str(report_path),
                "step_sessions": args.step,
                "caveats": [
                    "FA family frozen to the latest real report (selection-layer survivorship)",
                    "earnings flags from today's fetch (thinner at the oldest nights)",
                ],
                "nights": funnels,
                "records": records,
                "stats": stats,
                "errors": all_errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    closed = [r for r in records if r.get("status") in ("target", "stopped")]
    print(f"\nissued {stats['issued']} rows over {len(nights)} nights -> {out_path}")
    print(
        f"closed {len(closed)}: {stats['target']} target / {stats['stopped']} stopped"
        f" | hit_rate={stats['hit_rate']} total_r={stats['total_r']:+.2f}"
        f" avg_r={stats['avg_r'] if stats['avg_r'] is None else format(stats['avg_r'], '+.2f')}"
    )
    for tier in ("ticket", "watch"):
        ts = stats["by_tier"][tier]
        print(
            f"  {tier:6s}: issued={ts['issued']} target={ts['target']} stopped={ts['stopped']}"
            f" expired={ts['expired']} open={ts['open']} waiting={ts['waiting']}"
            f" total_r={ts['total_r']:+.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
