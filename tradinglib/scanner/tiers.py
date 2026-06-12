"""Nightly tiering: BH-FDR across the tested family + the watchlist tier.

The per-ticker tournament deflates every strategy by the whole config menu;
this module corrects the OTHER multiple-comparisons layer — selecting across
the ~80 ticker-stance tournaments run each night. ``p = 1 - DSR`` is a
Gaussian-approximation pseudo-p-value (see docs/methodology.md): it orders
and tiers candidates, it does not publish significance claims. Survivors that
fail the nightly FDR demote to the watchlist with the reason recorded — never
silently discarded — and the forward ledger tracks both tiers so the tiering
itself is empirically validated. Watch rows without levels (a survivor with
no actionable entry tonight) are report-only: there is nothing to simulate.
"""

from __future__ import annotations

from tradinglib.backtest.metrics import benjamini_hochberg_fdr

TIER_TICKET = "ticket"
TIER_WATCH = "watch"

# Per-setup entry windows (sessions). Default 5 == strategist.evaluate.ENTRY_WINDOW;
# the PEAD pair gets the drift's own persistence window (the detector accepts
# 3-15 days since the reaction bar, so a 5-session stale-trigger rule cuts the
# setup's thesis short).
ENTRY_WINDOWS: dict[str, int] = {"pead": 15, "pead_down": 15}
_DEFAULT_ENTRY_WINDOW = 5


def best_dsr(entry: dict) -> float | None:
    """Highest deflated Sharpe among an entry's verdicts; None without verdicts."""
    ds = [v["deflated_sharpe"] for v in entry.get("verdicts") or []]
    return max(ds) if ds else None


def apply_fdr(tournament: dict, alpha: float) -> tuple[dict[tuple[str, str], bool], float, int]:
    """Benjamini-Hochberg over every ticker-stance tested tonight.

    Returns ``(passed, threshold, family_size)`` where ``passed`` maps
    ``(stance, ticker)`` to the BH verdict on ``p = 1 - best DSR``.
    """
    keys: list[tuple[str, str]] = []
    pvalues: list[float] = []
    for stance in ("long", "short"):
        for entry in tournament.get(stance) or []:
            d = best_dsr(entry)
            if d is None:
                continue
            keys.append((stance, entry["ticker"]))
            pvalues.append(1.0 - d)
    rejected, threshold = benjamini_hochberg_fdr(pvalues, alpha)
    return dict(zip(keys, rejected, strict=True)), threshold, len(keys)


def _setup_watch_levels(setup: dict, stance: str) -> dict | None:
    """Detector trigger/stop turned into simulate-able levels (2R target).

    ``None`` when the 2R target is not a positive price (short risk can
    exceed trigger/2 on low-priced names) — the row stays report-only.
    """
    trigger, stop = float(setup["trigger_level"]), float(setup["stop_level"])
    risk = abs(stop - trigger)
    target = trigger + 2.0 * risk if stance == "long" else trigger - 2.0 * risk
    if target <= 0.0:
        return None
    return {"entry": trigger, "entry_type": "stop", "stop": stop, "target": target}


def build_watchlist(
    tournament: dict,
    candidates: list[dict],
    fdr_passed: dict[tuple[str, str], bool],
    *,
    watch_dsr_floor: float,
) -> dict[str, list[dict]]:
    """The watch tier, with the demotion reason recorded per row."""
    watchlist: dict[str, list[dict]] = {"long": [], "short": []}
    seen: set[tuple[str, str]] = set()

    for stance in ("long", "short"):
        for entry in tournament.get(stance) or []:
            if not entry.get("survivors"):
                continue
            key = (stance, entry["ticker"])
            winner = entry.get("winner")
            if winner is not None and fdr_passed.get(key, False):
                seen.add(key)  # a full ticket; not watch material
                continue
            if winner is not None:
                reason, levels, strategy = (
                    "failed nightly FDR",
                    winner["levels"],
                    winner["strategy"],
                )
            else:
                reason, levels = "no actionable entry tonight", None
                survived = [
                    v for v in entry["verdicts"] if v["strategy"] in set(entry["survivors"])
                ]
                if not survived:
                    continue  # malformed entry (survivors without verdicts) must not kill the scan
                strategy = max(survived, key=lambda v: v["deflated_sharpe"])["strategy"]
            seen.add(key)
            watchlist[stance].append(
                {
                    "ticker": entry["ticker"],
                    "stance": stance,
                    "strategy": strategy,
                    "tier": TIER_WATCH,
                    "tier_reason": reason,
                    "deflated_sharpe": best_dsr(entry),
                    "levels": levels,
                }
            )

    dsr_by_key = {
        (stance, e["ticker"]): best_dsr(e)
        for stance in ("long", "short")
        for e in tournament.get(stance) or []
    }
    for cand in candidates:
        stance = cand.get("cohort") or "long"
        if stance not in watchlist:
            raise ValueError(f"unknown cohort {stance!r} for {cand['ticker']}")
        key = (stance, cand["ticker"])
        if key in seen:
            continue
        d = dsr_by_key.get(key)
        if d is None or d < watch_dsr_floor:
            continue
        if not cand.get("setups"):
            continue
        setup = max(cand["setups"], key=lambda s: s["score"])
        watchlist[stance].append(
            {
                "ticker": cand["ticker"],
                "stance": stance,
                "strategy": f"setup:{setup['setup_type']}",
                "tier": TIER_WATCH,
                "tier_reason": "sub-threshold evidence",
                "entry_window": ENTRY_WINDOWS.get(setup["setup_type"], _DEFAULT_ENTRY_WINDOW),
                "deflated_sharpe": d,
                "levels": _setup_watch_levels(setup, stance),
            }
        )
    return watchlist
