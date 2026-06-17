"""Pooled cross-sectional setup certification.

Rare setups cannot clear the per-ticker 12-trade survival bar inside a
3-year window — PEAD fires a handful of times per name. This module pools
a setup TYPE's firings across the whole family on an as-of grid, scores
the pooled per-entry-date R series with the same deflated-Sharpe + BH-FDR
discipline the tournament uses, and certifies types — a parallel path to
ticket tier that adds power for low-frequency setups without weakening
the per-ticker bar. Same-entry-date firings aggregate to ONE observation:
cross-ticker trades on the same night are correlated, and counting them
separately would inflate the statistic.
"""

from __future__ import annotations

import pandas as pd

from tradinglib.backtest.metrics import benjamini_hochberg_fdr, compute_metrics
from tradinglib.scanner.config import ScanConfig
from tradinglib.scanner.setups import detect_all
from tradinglib.scanner.tiers import ENTRY_WINDOWS, _setup_watch_levels
from tradinglib.strategist.evaluate import ENTRY_WINDOW, simulate_ticket


def _naive(frame: pd.DataFrame) -> pd.DataFrame:
    """Bars on a tz-naive UTC index (mirrors the replay harness's normalization)."""
    idx = frame.index
    if getattr(idx, "tz", None) is not None:
        frame = frame.set_axis(idx.tz_convert("UTC").tz_localize(None))
    return frame


def sweep_firings(
    bars_by_ticker: dict[str, pd.DataFrame],
    *,
    setup_types: tuple[str, ...],
    stance: str,
    asof: pd.Timestamp,
    lookback_days: int,
    step_sessions: int,
    earnings_by_ticker: dict[str, pd.DatetimeIndex],
    setup_window_days: int = 450,
) -> list[dict]:
    """Every firing of the given setup types on a step-grid of past nights <= asof.

    tz-aware inputs are normalized to tz-naive UTC internally.
    """
    bars_by_ticker = {t: _naive(b) for t, b in bars_by_ticker.items()}
    if asof.tz is not None:
        asof = asof.tz_convert("UTC").tz_localize(None)
    calendars = [b.index for b in bars_by_ticker.values() if len(b)]
    if not calendars:
        return []
    sessions = calendars[0]
    for idx in calendars[1:]:
        sessions = sessions.union(idx)
    grid = sessions[(sessions >= asof - pd.Timedelta(days=lookback_days)) & (sessions <= asof)]
    out: list[dict] = []
    for night in grid[::step_sessions]:
        for ticker, bars in bars_by_ticker.items():
            window = bars.loc[
                (bars.index >= night - pd.Timedelta(days=setup_window_days)) & (bars.index <= night)
            ]
            assert len(window) == 0 or window.index.max() <= night
            for s in detect_all(
                window,
                stance=stance,
                earnings_datetimes=earnings_by_ticker.get(ticker, pd.DatetimeIndex([])),
            ):
                if s.setup_type not in setup_types:
                    continue
                out.append(
                    {
                        "date": night.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "stance": stance,
                        "setup_type": s.setup_type,
                        "entry_window": ENTRY_WINDOWS.get(s.setup_type, ENTRY_WINDOW),
                        "levels": _setup_watch_levels(
                            {"trigger_level": s.trigger_level, "stop_level": s.stop_level},
                            stance,
                        ),
                    }
                )
    return out


def pooled_r_series(scored: list[dict]) -> pd.Series:
    """Per-entry-date mean R over closed trades, chronological."""
    closed = [
        r
        for r in scored
        if r.get("status") in ("target", "stopped") and isinstance(r.get("r"), (int, float))
    ]
    if not closed:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(closed)
    series = frame.groupby("date")["r"].mean()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _deflated_sharpe(series: pd.Series, n_trials: int) -> float:
    """The tournament's deflation convention (compute_metrics) on a pooled R series.

    compute_metrics needs an equity curve only for max_drawdown, which
    certification ignores; a cumulative-R curve keeps the call well-formed.
    """
    equity = 1.0 + series.cumsum()
    return float(compute_metrics(series, equity, n_trials=n_trials)["deflated_sharpe"])


def certify(
    series_by_key: dict[tuple[str, str], pd.Series],
    *,
    n_trials: int,
    min_dates: int,
    fdr_alpha: float,
    dsr_threshold: float = 0.90,
) -> dict[tuple[str, str], dict]:
    """Certification verdict per (setup_type, stance) over pooled R series."""
    verdicts: dict[tuple[str, str], dict] = {}
    pvalues: list[float] = []
    keys: list[tuple[str, str]] = []
    for key, series in series_by_key.items():
        reasons: list[str] = []
        dsr = _deflated_sharpe(series, n_trials) if len(series) >= 2 else 0.0
        if len(series) < min_dates:
            reasons.append(f"n_dates {len(series)} < {min_dates}")
        if dsr < dsr_threshold:
            reasons.append(f"pooled_dsr {dsr:.2f} < {dsr_threshold:.2f}")
        verdicts[key] = {
            "pooled_dsr": dsr,
            "n_dates": len(series),
            "total_r": float(series.sum()),
            "reasons": reasons,
            "certified": False,
        }
        keys.append(key)
        pvalues.append(1.0 - dsr)
    rejected, _threshold = benjamini_hochberg_fdr(pvalues, fdr_alpha)
    for key, passed in zip(keys, rejected, strict=True):
        v = verdicts[key]
        if not passed and not v["reasons"]:
            v["reasons"].append("failed pooled FDR")
        # invariant: blocking reasons must NOT start with "failed" — that prefix marks informational-only
        v["certified"] = passed and not [r for r in v["reasons"] if not r.startswith("failed")]
    return verdicts


SETUP_TYPES: dict[str, tuple[str, ...]] = {
    "long": ("base_breakout", "ma_pullback", "pead"),
    "short": ("base_breakdown", "ma_rally_fade", "pead_down"),
}


def build_certification(
    bars_by_ticker: dict[str, pd.DataFrame],
    earnings_by_ticker: dict[str, pd.DatetimeIndex],
    *,
    asof: pd.Timestamp,
    config: ScanConfig,
) -> dict:
    """The weekly certification sidecar: every setup type x stance, pooled and judged.

    Simulation walks bars beyond each firing date with no upper bound: the
    caller must pre-slice ``bars_by_ticker`` to ``<= asof`` for a no-lookahead
    replay (the weekly prod sidecar relies on bars simply ending at asof).
    """
    series_by_key: dict[tuple[str, str], pd.Series] = {}
    # Each (setup_type, stance) key is a single pre-specified hypothesis: there is no
    # parameter search inside a key (sweep_firings only detects firings), so the honest
    # within-key trial count is 1 and the per-key statistic is the un-deflated PSR. The
    # cross-key multiplicity across the 6 keys is controlled in exactly one place —
    # certify()'s BH-FDR. Counting the 6 keys as n_trials here too would double-count
    # that multiplicity and over-deflate every per-key DSR (audit A1).
    n_trials = 1
    sim_errors = 0
    for stance, types in SETUP_TYPES.items():
        firings = sweep_firings(
            bars_by_ticker,
            setup_types=types,
            stance=stance,
            asof=asof,
            lookback_days=config.pooled_lookback_days,
            step_sessions=config.pooled_step_sessions,
            earnings_by_ticker=earnings_by_ticker,
        )
        scored = []
        for row in firings:
            if not row.get("levels"):
                continue  # unattainable 2R target: report-only, nothing to simulate
            try:
                scored.append(
                    {
                        **row,
                        **simulate_ticket(
                            row,
                            bars_by_ticker[row["ticker"]],
                            asof=row["date"],
                            entry_window=row["entry_window"],
                        ),
                    }
                )
            except Exception:
                sim_errors += 1
                continue
        for setup_type in types:
            series_by_key[(setup_type, stance)] = pooled_r_series(
                [s for s in scored if s["setup_type"] == setup_type]
            )
    verdicts = certify(
        series_by_key,
        n_trials=n_trials,
        min_dates=config.pooled_min_dates,
        fdr_alpha=config.fdr_alpha,
    )
    return {
        "built_asof": asof.strftime("%Y-%m-%d"),
        "lookback_days": config.pooled_lookback_days,
        "sim_errors": sim_errors,
        "verdicts": {f"{t}:{s}": v for (t, s), v in verdicts.items()},
    }
