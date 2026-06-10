"""Scan configuration shared by the pipeline stages and the CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScanConfig:
    """Knobs for one scanner run.

    ``limit`` truncates the universe for smoke runs; ``fa_keep`` is the
    long shortlist size after the fundamental gate and ``short_keep`` the
    short-candidate slate (bottom of the FA ranking, 0 disables); ``top``
    caps the ranked watchlist in the report. ``universe`` selects the
    constituent list (``"russell1000"`` or ``"sp500"``). ``skip_strategies``
    disables the per-ticker strategy tournament over the FA candidates;
    ``tournament_lookback_days`` is the bar history it loads (~3y+ so the
    378-bar anchored train leaves several OOS quarters).
    """

    fa_keep: int = 40
    short_keep: int = 40
    top: int = 15
    limit: int | None = None
    refresh: bool = False
    skip_llm: bool = False
    edgar_enrich: bool = True
    lookback_days: int = 450
    earnings_warn_days: int = 14
    universe: str = "russell1000"
    skip_strategies: bool = False
    tournament_lookback_days: int = 1200
