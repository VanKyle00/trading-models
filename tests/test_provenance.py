"""Tests for the shared point-in-time / survivorship provenance (audit B-series)."""

from __future__ import annotations

from tradinglib.provenance import (
    BIASED_UPPER_BOUND_BANNER,
    FUNDAMENTALS_RESTATED,
    MEMBERSHIP_SURVIVORSHIP,
    Provenance,
    from_reasons,
    honesty_summary,
    merge,
)


def test_from_reasons_sets_leak_true_when_any_reason() -> None:
    p = from_reasons(MEMBERSHIP_SURVIVORSHIP)
    assert isinstance(p, Provenance)
    assert p.leak is True and p.reasons == (MEMBERSHIP_SURVIVORSHIP,)


def test_from_reasons_no_reason_is_clean() -> None:
    p = from_reasons()
    assert p.leak is False and p.reasons == ()


def test_merge_unions_reasons_and_dedups_and_sorts() -> None:
    a = from_reasons(MEMBERSHIP_SURVIVORSHIP)
    b = from_reasons(FUNDAMENTALS_RESTATED, MEMBERSHIP_SURVIVORSHIP)
    m = merge(a, b)
    assert m.leak is True
    assert m.reasons == (FUNDAMENTALS_RESTATED, MEMBERSHIP_SURVIVORSHIP)  # sorted, deduped


def test_merge_of_clean_is_clean() -> None:
    assert merge(from_reasons(), from_reasons()).leak is False


def test_honesty_summary_counts_leaked_records() -> None:
    recs = [{"leak": True}, {"leak": False}, {"leak": True}, {}]
    assert honesty_summary(recs) == {"leaked": True, "leaked_records": 2, "honest_records": 2}


def test_banner_mentions_relative_only() -> None:
    assert "RELATIVE" in BIASED_UPPER_BOUND_BANNER.upper()
