"""No-lookahead historical replay of the technical scan funnel.

Examples:
    uv run python scripts/backfill_scan.py --smoke          # one night, timed
    uv run python scripts/backfill_scan.py                  # full replay
    uv run python scripts/backfill_scan.py --days 300 --step 5
    uv run python scripts/backfill_scan.py --days 300 --step 5 --regime-gate   # gated A/B arm
    uv run python scripts/backfill_scan.py --days 300 --step 5 --pooled        # pooled A/B arm

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
from tradinglib.provenance import BIASED_UPPER_BOUND_BANNER, honesty_summary
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.ledger import _stats
from tradinglib.scanner.pooled import build_certification
from tradinglib.scanner.regime import gate_reason, regime_state
from tradinglib.scanner.setups import detect_all
from tradinglib.scanner.tiers import apply_fdr, build_watchlist
from tradinglib.strategist.evaluate import ENTRY_WINDOW, simulate_ticket
from tradinglib.tournament.run import run_tournament

_BENCHMARK = "SPY"


def _campaign_active(record: dict, bars: pd.DataFrame, night: pd.Timestamp) -> bool:
    """Whether a previously issued (and fully simulated) row is still waiting/open at `night`.

    Derived from the one full-history simulation instead of re-simulating per
    night: an unfilled entry stays a live campaign until its entry window of
    post-issue sessions has elapsed; a filled one until its exit date.
    """
    issue = pd.Timestamp(record["date"])
    if record.get("status") == "error":
        return False  # error rows never suppress (mirrors prod: errors fail open)
    if record.get("entry_date") is None:
        window = int(record.get("entry_window", ENTRY_WINDOW))
        sessions_elapsed = int(((bars.index > issue) & (bars.index <= night)).sum())
        return sessions_elapsed < window
    if record.get("exit_date") is None:
        return True  # open through the end of data
    return night < pd.Timestamp(record["exit_date"])


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


def load_archived_families(base: Path) -> list[tuple[str, dict[str, list[str]]]]:
    """(date, family) per real report, oldest first."""
    out: list[tuple[str, dict[str, list[str]]]] = []
    if not base.exists():
        return out
    for d in sorted(p.name for p in base.iterdir() if (p / "report.json").exists()):
        try:
            out.append((d, _family(base / d / "report.json")))
        except (KeyError, json.JSONDecodeError):
            continue  # pre-FA-format or corrupt report: skip
    return out


def family_for_night(
    families: list[tuple[str, dict[str, list[str]]]], night: pd.Timestamp
) -> tuple[dict[str, list[str]], str]:
    """Newest archived family dated <= night; oldest family + 'fallback' before coverage.

    Assumes non-empty ``families`` — the caller aborts before resolving when no
    archived reports exist.
    """
    night_str = night.strftime("%Y-%m-%d")
    chosen = None
    for date, fam in families:
        if date <= night_str:
            chosen = (fam, date)
    if chosen is not None:
        return chosen
    return families[0][1], "fallback"


def point_in_time_status(family_mode: str, family_source: str | None) -> tuple[str, bool]:
    """(label, leak) for a night's family provenance — a machine-enforced flag.

    The replay's FA family is a SELECTION layer: which 40+40 names are scored.
    - frozen   -> ("frozen", True): today's FA report back-projected onto every
      night, chosen on current/restated fundamentals = selection-layer leak.
    - archived served by a real report dated <= night -> ("archived", False):
      point-in-time honest.
    - archived before coverage (reuses the oldest report) -> ("fallback", True).

    Downstream consumers must suppress or clearly segregate absolute per-tier R
    when leak is True (it is a biased upper bound, not an expected edge).
    """
    if family_mode == "archived" and family_source not in (None, "fallback"):
        return "archived", False
    if family_mode == "archived" and family_source == "fallback":
        return "fallback", True
    return "frozen", True


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


def _promote_certified(
    watchlist: dict[str, list[dict]],
    certification: dict,
    date: str,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Promote pooled-certified setup watch rows to ticket tier (replays the pipeline).

    Mirrors tradinglib.scanner.pipeline: a setup:* watch row with levels whose
    (setup_type, stance) verdict is certified leaves the watchlist and becomes a
    ticket-tier issued row. Returns (watchlist sans promoted, promoted issued rows).

    Shape divergence from prod (deliberate): the replay row drops
    ``pooled_evidence`` and ``deflated_sharpe``; the artifact scores from
    ``tier_reason`` + simulated R rather than replaying option-chain/evidence fields.
    """
    verdicts = certification.get("verdicts", {})
    promoted: list[dict] = []
    kept_by_stance: dict[str, list[dict]] = {}
    for stance in ("long", "short"):
        kept = []
        for row in watchlist[stance]:
            key = f"{row['strategy'].removeprefix('setup:')}:{stance}"
            verdict = verdicts.get(key)
            if (
                row["strategy"].startswith("setup:")
                and row.get("levels")
                and verdict
                and verdict.get("certified")
            ):
                promoted.append(
                    {
                        "date": date,
                        "ticker": row["ticker"],
                        "stance": stance,
                        "strategy": row["strategy"],
                        "tier": "ticket",
                        "tier_reason": "pooled-certified",
                        "entry_window": int(row.get("entry_window", ENTRY_WINDOW)),
                        "levels": row["levels"],
                    }
                )
            else:
                kept.append(row)
        kept_by_stance[stance] = kept
    return kept_by_stance, promoted


def run_night(
    asof: pd.Timestamp,
    family: dict[str, list[str]],
    bars_by_ticker: dict[str, pd.DataFrame],
    earnings_by_ticker: dict[str, pd.DatetimeIndex],
    config: ScanConfig,
    certification: dict | None = None,
    sessions_by_ticker: dict[str, list[str]] | None = None,
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
    regime = regime_state(bench_close)
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
                    earnings_sessions=(
                        sessions_by_ticker.get(ticker) if sessions_by_ticker else None
                    ),
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
        tournament,
        candidates,
        fdr_passed,
        watch_dsr_floor=config.watch_dsr_floor,
        setup_score_floors=config.setup_score_floors,
    )

    date = asof.strftime("%Y-%m-%d")
    # Pooled-certified promotion: setup watch rows of certified types graduate to
    # ticket tier and leave the watchlist (mirrors the pipeline). ORDERING NOTE:
    # prod orders promotion -> cooldown -> regime; this replay runs regime (below,
    # in run_night) before cooldown (in main). The two filters are independent
    # predicates, so the SET of surviving rows is identical either order; only
    # doubly-blocked attribution differs (see the artifact's regime_blocked caveat).
    promoted: list[dict] = []
    if certification and config.pooled_certification:
        watchlist, promoted = _promote_certified(watchlist, certification, date)

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
                    "entry_window": int(row.get("entry_window", ENTRY_WINDOW)),
                    **{k: row[k] for k in ("ticker", "stance", "strategy", "tier", "levels")},
                }
            )
    # promoted rows join the issued set BEFORE the regime gate, so the existing
    # gate applies to them; cooldown (main) applies via their ticket-tier keys.
    pooled_promoted = len(promoted)
    issued.extend(promoted)

    regime_blocked: list[dict] = []
    if config.regime_gate:
        regime_blocked = [
            row for row in issued if gate_reason(row["strategy"], row["stance"], regime) is not None
        ]
        issued = [row for row in issued if row not in regime_blocked]

    funnel = {
        "date": date,
        "tournaments": sum(len(v) for v in tournament.values()),
        "survivors": sum(1 for v in tournament.values() for e in v if e["survivors"]),
        "fdr_family": fdr_family,
        "fdr_passed": sum(1 for v in fdr_passed.values() if v),
        "tickets": sum(1 for r in issued if r["tier"] == "ticket"),
        "watch": sum(1 for r in issued if r["tier"] == "watch"),
        # promotions at issuance (pre-suppression) — mirrors prod's funnel semantics
        "pooled_promoted": pooled_promoted,
        "regime_trend": regime["trend"],
        "regime_blocked": len(regime_blocked),
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
    parser.add_argument(
        "--regime-gate",
        action="store_true",
        help="apply the meanrev-against-trend gate at issuance (A/B arm)",
    )
    parser.add_argument(
        "--family-mode",
        choices=("frozen", "archived"),
        default="frozen",
        help="frozen: one family from --report; archived: per-night family from archived reports",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="apply pooled-certified promotion from a per-night as-of certification (A/B arm)",
    )
    args = parser.parse_args(argv)

    config = ScanConfig(regime_gate=args.regime_gate)
    today = pd.Timestamp(datetime.now(UTC).strftime("%Y-%m-%d"))
    replay_start = today - pd.Timedelta(days=args.days)
    bar_start = (replay_start - pd.Timedelta(days=config.tournament_lookback_days)).strftime(
        "%Y-%m-%d"
    )

    if args.family_mode == "archived":
        if args.report is not None:
            print("--report is ignored in archived mode (families come from archived reports)")
        report_path = None
        families = load_archived_families(processed_dir("scans"))
        if not families:
            print("no archived FA reports under data/processed/scans -- nothing to replay")
            return 1
        tickers = sorted(
            {t for _, fam in families for stance in ("long", "short") for t in fam[stance]}
        )
        print(
            f"families: {len(families)} archived reports "
            f"({families[0][0]} .. {families[-1][0]}), union {len(tickers)} tickers"
        )
    else:
        report_path = args.report
        if report_path is None:
            dates = sorted(
                p.name for p in processed_dir("scans").iterdir() if (p / "report.json").exists()
            )
            report_path = processed_dir("scans") / dates[-1] / "report.json"
        family = _family(report_path)
        tickers = sorted(set(family["long"]) | set(family["short"]))
        print(
            f"family: {len(family['long'])} long / {len(family['short'])} short from {report_path}"
        )

    bars_by_ticker: dict[str, pd.DataFrame] = {}
    earnings_by_ticker: dict[str, pd.DatetimeIndex] = {}
    sessions_by_ticker: dict[str, list[str]] = {}
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
            sessions_by_ticker[ticker] = list(earnings["session"])
        except Exception:
            earnings_by_ticker[ticker] = pd.DatetimeIndex([])
            sessions_by_ticker[ticker] = []
    print(f"bars loaded: {len(bars_by_ticker)}/{len(tickers) + 1}")

    calendar = bars_by_ticker[_BENCHMARK].loc[lambda b: b.index >= replay_start].index
    nights = list(calendar[:: args.step])
    if args.smoke:
        nights = nights[:1]
    print(f"replaying {len(nights)} nights ({nights[0].date()} .. {nights[-1].date()})")

    funnels: list[dict] = []
    records: list[dict] = []
    all_errors: list[str] = []
    cert_cache: dict[str, dict] = {}
    for i, asof in enumerate(nights, 1):
        t0 = time.monotonic()
        family_suffix = ""
        family_source = None
        if args.family_mode == "archived":
            family, family_source = family_for_night(families, asof)
        certification = None
        if args.pooled:
            # archived mode: the cert pools over the union of all archived
            # families (superset evidence); prod uses last night's family
            week = f"{asof.isocalendar().year}-{asof.isocalendar().week:02d}"
            if week not in cert_cache:
                t_cert = time.monotonic()
                sliced = {
                    t: _slice(b, asof, config.pooled_lookback_days + 450)
                    for t, b in bars_by_ticker.items()
                    if t != _BENCHMARK
                }
                cert_cache[week] = build_certification(
                    sliced,
                    earnings_by_ticker,
                    asof=asof,
                    config=config,
                    sessions_by_ticker=sessions_by_ticker,
                )
                certified = sum(1 for v in cert_cache[week]["verdicts"].values() if v["certified"])
                print(
                    f"  certification {week}: {certified}/6 certified,"
                    f" sim_errors={cert_cache[week]['sim_errors']}"
                    f" ({time.monotonic() - t_cert:.0f}s)",
                    flush=True,
                )
            certification = cert_cache[week]
        funnel, issued, errors = run_night(
            asof,
            family,
            bars_by_ticker,
            earnings_by_ticker,
            config,
            certification=certification,
            sessions_by_ticker=sessions_by_ticker,
        )
        night_pit, night_leak = point_in_time_status(args.family_mode, family_source)
        funnel["point_in_time"], funnel["leak"] = night_pit, night_leak
        if args.family_mode == "archived":
            funnel["family_source"] = family_source
            family_suffix = f" family={family_source}"
        # prod's re-issue cooldown, replayed: a campaign still waiting/open at
        # this night suppresses tonight's same (ticker, stance, strategy, tier)
        active = {
            (r["ticker"], r["stance"], r["strategy"], r["tier"])
            for r in records
            if _campaign_active(r, bars_by_ticker[r["ticker"]], asof)
        }
        suppressed = [
            row
            for row in issued
            if (row["ticker"], row["stance"], row["strategy"], row["tier"]) in active
        ]
        issued = [row for row in issued if row not in suppressed]
        funnel["suppressed"] = len(suppressed)
        funnel["tickets"] -= sum(1 for r in suppressed if r["tier"] == "ticket")
        funnel["watch"] -= sum(1 for r in suppressed if r["tier"] == "watch")
        for row in issued:
            record = dict(row)
            record["point_in_time"], record["leak"] = night_pit, night_leak
            if args.family_mode == "archived":
                record["family_source"] = family_source
            try:
                record.update(
                    simulate_ticket(
                        row,
                        bars_by_ticker[row["ticker"]],
                        asof=row["date"],
                        entry_window=int(row.get("entry_window", ENTRY_WINDOW)),
                    )
                )
            except Exception as exc:
                record.update(status="error", error=str(exc))
            records.append(record)
        funnels.append(funnel)
        all_errors.extend(errors)
        pooled_suffix = f"pooled_promoted={funnel['pooled_promoted']} " if args.pooled else ""
        print(
            f"[{i}/{len(nights)}] {funnel['date']}: survivors={funnel['survivors']} "
            f"fdr_passed={funnel['fdr_passed']} tickets={funnel['tickets']} "
            f"watch={funnel['watch']} suppressed={funnel['suppressed']} "
            f"{pooled_suffix}"
            f"regime_blocked={funnel['regime_blocked']} "
            f"best_dsr={funnel['best_dsr'] or 0:.3f} "
            f"errors={funnel['errors']}{family_suffix} ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )

    stats = {
        **_stats(records),
        "by_tier": {t: _stats([r for r in records if r["tier"] == t]) for t in ("ticket", "watch")},
        "by_stance": {
            s: _stats([r for r in records if r["stance"] == s]) for s in ("long", "short")
        },
        "pooled_certified": _stats(
            [r for r in records if r.get("tier_reason") == "pooled-certified"]
        ),
    }
    honesty = {
        **honesty_summary(records),
        "point_in_time_nights": sum(1 for f in funnels if not f["leak"]),
        "total_nights": len(funnels),
    }
    out_path = args.out or (
        processed_dir("backfill")
        / f"replay_{nights[0].strftime('%Y%m%d')}_{nights[-1].strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    family_caveat = (
        "FA families resolved per-night from archived reports; nights before coverage reuse "
        "the oldest report (family_source='fallback')"
        if args.family_mode == "archived"
        else "FA family frozen to the latest real report (selection-layer survivorship)"
    )
    out_path.write_text(
        json.dumps(
            {
                "generated_asof": today.strftime("%Y-%m-%d"),
                "family_report": str(report_path) if report_path is not None else None,
                "family_mode": args.family_mode,
                "step_sessions": args.step,
                "regime_gate": args.regime_gate,
                "pooled": args.pooled,
                "caveats": [
                    family_caveat,
                    "earnings flags from today's fetch (thinner at the oldest nights)",
                    "regime_blocked attribution differs from prod: replay gates before the "
                    "cooldown (prod: after, so doubly-blocked rows read 'open campaign') and "
                    "never sees report-only watch rows prod would gate; issued set unaffected. "
                    "Pooled-certified promotions are gated before cooldown too (prod: after); "
                    "same independent-predicate argument, so the issued set is unaffected",
                ],
                "honesty": honesty,
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
        f"point-in-time-honest nights: {honesty['point_in_time_nights']}/{honesty['total_nights']}"
        f" | leaked records: {honesty['leaked_records']}/{stats['issued']}"
    )
    if honesty["leaked"]:
        print(f"  {BIASED_UPPER_BOUND_BANNER}")
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
    if args.family_mode == "archived":
        honest = [f for f in funnels if f.get("family_source", "fallback") != "fallback"]
        print(f"nights with archived (point-in-time) families: {len(honest)}/{len(funnels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
