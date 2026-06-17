"""Machine-enforced point-in-time / survivorship provenance (audit B-series).

One source of truth for the leak flag + BIASED-UPPER-BOUND banner that the
backfill replay (``scripts/backfill_scan.py``) and the live pipeline both use, so
absolute backtest/replay numbers are always honestly flagged. ``leak=True`` marks
a number whose evidence is non-point-in-time / survivor-biased — a biased upper
bound, trustworthy only as a RELATIVE A/B delta, not as an absolute edge. The
``reasons`` tuple names which axes are dirty so the PIT arms (EDGAR-PIT
fundamentals, unadjusted prices) can flip a record's reasons off as they land.
"""

from __future__ import annotations

from dataclasses import dataclass

MEMBERSHIP_SURVIVORSHIP = "membership-survivorship"
FUNDAMENTALS_RESTATED = "fundamentals-restated"
PRICE_RESTATED = "price-restated"

BIASED_UPPER_BOUND_BANNER = (
    "*** BIASED UPPER-BOUND DIAGNOSTIC: absolute R / hit-rate are inflated by "
    "non-point-in-time / survivor-biased evidence. Trust only RELATIVE A/B deltas. ***"
)


@dataclass(frozen=True)
class Provenance:
    """Whether a number's evidence leaks, and on which axes."""

    leak: bool
    reasons: tuple[str, ...]


def from_reasons(*reasons: str) -> Provenance:
    """Build a Provenance from leak reasons; ``leak`` is True iff any reason is given."""
    uniq = tuple(sorted(set(reasons)))
    return Provenance(leak=bool(uniq), reasons=uniq)


def merge(*provs: Provenance) -> Provenance:
    """Union of reasons across provenances; ``leak`` is True iff any input leaks."""
    reasons: set[str] = set()
    for p in provs:
        reasons.update(p.reasons)
    return from_reasons(*reasons)


def honesty_summary(records: list[dict]) -> dict:
    """Aggregate per-record ``leak`` flags into a machine-readable honesty block."""
    leaked = sum(1 for r in records if r.get("leak"))
    return {"leaked": leaked > 0, "leaked_records": leaked, "honest_records": len(records) - leaked}
