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

from tradinglib.scanner.setups import detect_all
from tradinglib.scanner.tiers import ENTRY_WINDOWS, _setup_watch_levels
from tradinglib.strategist.evaluate import ENTRY_WINDOW


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
