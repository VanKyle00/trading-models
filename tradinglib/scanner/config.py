"""Scan configuration shared by the pipeline stages and the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    ``account_size`` x ``risk_per_trade_pct`` is the per-ticket risk budget
    the ticket playbook sizes against (the CSP is capital-sized).
    ``fdr_alpha`` drives the nightly Benjamini-Hochberg pass over the tested family;
    survivors that fail it demote to the watchlist, never silently dropped.
    ``watch_dsr_floor`` is the minimum best-verdict DSR for a setup-fired ticker
    to make the watchlist.
    ``setup_score_floors`` maps setup_type to a minimum score for the sub-threshold
    watch path; EMPTY by default — a floor requires live-ledger evidence, not replay
    tuning (see the umbrella plan).
    ``regime_gate`` gates meanrev-against-trend issuance; OFF until replay + live
    evidence (regime is always reported regardless).
    ``pooled_certification`` promotes watch rows of pooled-certified setup types
    to ticket tier from the weekly certification sidecar; ``pooled_min_dates`` /
    ``pooled_lookback_days`` / ``pooled_step_sessions`` shape the pooled sweep
    (entry-date observations required, calendar lookback, grid stride).
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
    account_size: float = 100_000.0
    risk_per_trade_pct: float = 0.01
    fdr_alpha: float = 0.10
    watch_dsr_floor: float = 0.5
    setup_score_floors: dict[str, float] = field(default_factory=dict)
    regime_gate: bool = False
    pooled_certification: bool = True
    pooled_min_dates: int = 20
    pooled_lookback_days: int = 750
    pooled_step_sessions: int = 5
