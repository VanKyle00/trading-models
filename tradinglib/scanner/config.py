"""Scan configuration shared by the pipeline stages and the CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScanConfig:
    """Knobs for one scanner run.

    ``limit`` truncates the universe for smoke runs; ``fa_keep`` is the
    shortlist size after the fundamental gate; ``top`` caps the ranked
    watchlist in the report.
    """

    fa_keep: int = 40
    top: int = 15
    limit: int | None = None
    refresh: bool = False
    skip_llm: bool = False
    edgar_enrich: bool = True
    lookback_days: int = 450
    earnings_warn_days: int = 14
