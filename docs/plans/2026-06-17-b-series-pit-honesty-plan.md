# B-series Point-in-Time Honesty Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make absolute backtest/replay numbers honest (machine-flagged) and remove the fixable PIT leaks (EDGAR fundamentals restatement, price restatement), per `docs/plans/2026-06-17-b-series-pit-honesty.md`.

**Architecture:** A shared `provenance` module carries a `leak`/`reasons` flag from the data layer through pipeline → ledger → pooled-cert → prints. Membership survivorship is permanently banner-flagged (no free PIT). EDGAR fundamentals and prices gain real as-of/unadjusted paths that are additive (`asof=None` / adjusted columns = byte-identical to today).

**Tech Stack:** Python 3.12, pandas, pytest, ruff, mypy; yfinance + SEC EDGAR (free data only).

## Global Constraints

- Verify each change against the full gate before commit: `python -m pytest -q`, `python -m ruff check .`, `python -m ruff format --check .`, `python -m mypy tradinglib`.
- `python` = `.venv/Scripts/python.exe`.
- ADDITIVE ONLY: `asof=None` routes the existing fundamentals path byte-identical; the adjusted price columns indicators read are never altered; the `leak` flag is metadata + banner and NEVER changes an R/hit-rate value.
- ruff: line-length 100, `E501` ignored; run `ruff format` after edits.
- One feature branch `feat/b-series-pit-honesty`; one commit per task.
- Banner string is defined ONCE in `tradinglib/provenance.py` and imported everywhere (DRY).

---

## Stage 1 — Provenance module + backfill_scan refactor

### Task 1: `tradinglib/provenance.py`

**Files:**
- Create: `tradinglib/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces:
  - `class Provenance(frozen dataclass)` with `leak: bool`, `reasons: tuple[str, ...]`
  - `MEMBERSHIP_SURVIVORSHIP = "membership-survivorship"`, `FUNDAMENTALS_RESTATED = "fundamentals-restated"`, `PRICE_RESTATED = "price-restated"` (str consts)
  - `def merge(*provs: Provenance) -> Provenance` — union of reasons, `leak = any`
  - `def from_reasons(*reasons: str) -> Provenance` — `leak = bool(reasons)`
  - `def honesty_summary(records: list[dict]) -> dict` — `{"leaked": bool, "leaked_records": int, "honest_records": int}` reading each record's `"leak"` key
  - `BIASED_UPPER_BOUND_BANNER: str` — one canonical banner line

- [ ] **Step 1: Write failing tests**

```python
# tests/test_provenance.py
from tradinglib.provenance import (
    Provenance, merge, from_reasons, honesty_summary,
    MEMBERSHIP_SURVIVORSHIP, FUNDAMENTALS_RESTATED, BIASED_UPPER_BOUND_BANNER,
)

def test_from_reasons_sets_leak_true_when_any_reason():
    p = from_reasons(MEMBERSHIP_SURVIVORSHIP)
    assert p.leak is True and p.reasons == (MEMBERSHIP_SURVIVORSHIP,)

def test_from_reasons_no_reason_is_clean():
    p = from_reasons()
    assert p.leak is False and p.reasons == ()

def test_merge_unions_reasons_and_dedups_and_sorts():
    a = from_reasons(MEMBERSHIP_SURVIVORSHIP)
    b = from_reasons(FUNDAMENTALS_RESTATED, MEMBERSHIP_SURVIVORSHIP)
    m = merge(a, b)
    assert m.leak is True
    assert m.reasons == (FUNDAMENTALS_RESTATED, MEMBERSHIP_SURVIVORSHIP)  # sorted, deduped

def test_merge_of_clean_is_clean():
    assert merge(from_reasons(), from_reasons()).leak is False

def test_honesty_summary_counts_leaked_records():
    recs = [{"leak": True}, {"leak": False}, {"leak": True}, {}]
    s = honesty_summary(recs)
    assert s == {"leaked": True, "leaked_records": 2, "honest_records": 2}

def test_banner_mentions_relative_only():
    assert "RELATIVE" in BIASED_UPPER_BOUND_BANNER.upper()
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_provenance.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# tradinglib/provenance.py
"""Machine-enforced point-in-time / survivorship provenance (audit B-series).

One source of truth for the leak flag + BIASED-UPPER-BOUND banner that the
backfill replay (scripts/backfill_scan.py) and the live pipeline both use, so
absolute backtest/replay numbers are always honestly flagged. `leak=True` marks
a number whose evidence is non-PIT / survivor-biased — a biased upper bound,
trustworthy only as a RELATIVE A/B delta, not as an absolute edge.
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
    leak: bool
    reasons: tuple[str, ...]


def from_reasons(*reasons: str) -> Provenance:
    uniq = tuple(sorted(set(reasons)))
    return Provenance(leak=bool(uniq), reasons=uniq)


def merge(*provs: Provenance) -> Provenance:
    reasons: set[str] = set()
    for p in provs:
        reasons.update(p.reasons)
    return from_reasons(*reasons)


def honesty_summary(records: list[dict]) -> dict:
    leaked = sum(1 for r in records if r.get("leak"))
    return {"leaked": leaked > 0, "leaked_records": leaked, "honest_records": len(records) - leaked}
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest tests/test_provenance.py -q` → PASS.
- [ ] **Step 5: Gate + commit** — run the full gate; `git commit -m "feat(provenance): shared leak/honesty module (audit B-series)"`.

### Task 2: Refactor `backfill_scan.py` onto `provenance` (no behavior change)

**Files:**
- Modify: `scripts/backfill_scan.py` (`point_in_time_status`, `honesty_summary`, banner literal)
- Test: existing `tests/test_backfill_determinism.py` must stay green.

**Interfaces:**
- Consumes: `tradinglib.provenance.{honesty_summary, BIASED_UPPER_BOUND_BANNER, MEMBERSHIP_SURVIVORSHIP}`
- `point_in_time_status` stays in backfill_scan (family-selection specific) but returns reasons drawn from the provenance consts; `honesty_summary` and the banner now come from `provenance`.

- [ ] **Step 1:** Run existing `tests/test_backfill_determinism.py -q` → PASS (baseline).
- [ ] **Step 2:** Replace `backfill_scan.honesty_summary` with an import of `provenance.honesty_summary`; replace the inline banner string with `provenance.BIASED_UPPER_BOUND_BANNER`; keep `point_in_time_status` but have its frozen/fallback branches return `MEMBERSHIP_SURVIVORSHIP`-style reasons. Remove the now-dead local definitions.
- [ ] **Step 3:** Run `tests/test_backfill_determinism.py -q` + any backfill tests → PASS (behavior unchanged; banner text may shift — update the assertion if a test pins the exact banner string).
- [ ] **Step 4: Gate + commit** — `git commit -m "refactor(backfill): use shared provenance module (no behavior change)"`.

---

## Stage 2 — Membership banner + leak threading (closes #89/#90 at the honesty level)

### Task 3: Universe loaders stamp a membership-survivorship provenance

**Files:**
- Modify: `tradinglib/loaders/universe/sp500.py`, `tradinglib/loaders/universe/russell1000.py` (attach provenance to the returned frame's `attrs`)
- Test: `tests/test_universe_loader.py`, `tests/test_russell1000_loader.py`

**Interfaces:**
- Produces: returned DataFrame carries `df.attrs["provenance"] = from_reasons(MEMBERSHIP_SURVIVORSHIP)` (in addition to the existing `attrs["snapshot"]`).

- [ ] **Step 1: Failing test** (sp500 + russell1000):

```python
from tradinglib.provenance import MEMBERSHIP_SURVIVORSHIP
def test_sp500_frame_carries_membership_provenance(monkeypatch, ...):
    df = get_sp500_constituents()
    prov = df.attrs["provenance"]
    assert prov.leak is True and MEMBERSHIP_SURVIVORSHIP in prov.reasons
```
(reuse each test file's existing fixture/monkeypatch for the scrape.)

- [ ] **Step 2:** Run → FAIL (no `provenance` attr).
- [ ] **Step 3:** In each loader, after building the canonical frame, set `out.attrs["provenance"] = from_reasons(MEMBERSHIP_SURVIVORSHIP)` on every return path (cache-hit, fresh, fallback).
- [ ] **Step 4:** Run both loader test files → PASS.
- [ ] **Step 5: Gate + commit** — `git commit -m "feat(universe): stamp membership-survivorship provenance (audit #89)"`.

### Task 4: Thread leak/reasons onto pipeline records + pooled-cert caveat

**Files:**
- Modify: `tradinglib/scanner/pipeline.py` (origin: compute per-run/per-name `Provenance` in `run_scan`; attach `leak`/`reasons` to issued + pooled-promoted records, `~:306-323`)
- Modify: `tradinglib/scanner/report.py` (surface the banner via `honesty_summary`)
- Test: `tests/test_scanner_pipeline*.py` (find the existing pipeline test) + a new assertion

**Interfaces:**
- Consumes: `df.attrs["provenance"]` from Task 3; `provenance.{merge, from_reasons, honesty_summary, BIASED_UPPER_BOUND_BANNER, FUNDAMENTALS_RESTATED}`
- Produces: every issued/promoted record dict has `"leak": bool` and `"reasons": list[str]`; pooled-promoted tickets (`pipeline.py:306-323`) carry the survivor-evidence caveat. Default today: `leak=True` with `MEMBERSHIP_SURVIVORSHIP` (always) and `FUNDAMENTALS_RESTATED` (until Stage 3 feeds EDGAR-PIT).

- [ ] **Step 1: Failing test** — run a small scan (existing pipeline-test fixture) and assert each issued record has `record["leak"] is True` and `MEMBERSHIP_SURVIVORSHIP in record["reasons"]`; assert a pooled-promoted ticket carries the caveat reasons.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** In `run_scan`, build `run_prov = merge(universe.attrs["provenance"], from_reasons(FUNDAMENTALS_RESTATED))`; stamp `record["leak"]=run_prov.leak`, `record["reasons"]=list(run_prov.reasons)` on issued records and on the pooled promotion block (`pipeline.py:306-323`). In `report.py`, when `honesty_summary(records)["leaked"]`, prepend `BIASED_UPPER_BOUND_BANNER` to the absolute-number section.
- [ ] **Step 4:** Run pipeline tests → PASS; assert NO R/hit-rate value changed (only `leak`/`reasons` keys added).
- [ ] **Step 5: Gate + commit** — `git commit -m "feat(pipeline): thread leak/honesty flag + pooled-cert caveat (audit #89/#90)"`.

---

## Stage 3 — EDGAR-PIT fundamentals (drops `fundamentals-restated` for covered names)

### Task 5: As-of `get_quarterly_trends`

**Files:**
- Modify: `tradinglib/loaders/fundamentals/edgar.py` (`_quarterly_values`, `get_quarterly_trends`, cache key)
- Test: `tests/test_fundamentals_edgar.py`

**Interfaces:**
- Consumes: SEC `companyfacts` JSON `facts.us-gaap.<tag>.units.<unit>[]` entries (`val`/`filed`/`start`/`end`/`accn`/`form`)
- Produces: `get_quarterly_trends(cik, *, asof: pd.Timestamp | None = None, refresh=False, client=None) -> dict[str, float]` — same 4 keys. `asof=None` ⇒ existing frame path UNCHANGED. `asof` set ⇒ as-of `units` path; cache keyed by `asof or today`.

- [ ] **Step 1: Failing tests** (use a fixture with two filings for the same quarter — an original and a later restatement — plus a future filing):

```python
def test_asof_picks_latest_filing_on_or_before_asof_and_ignores_future():
    facts = _facts_fixture()  # CY2023Q1 filed 2023-05-01 val=100; restated filed 2024-02-01 val=130;
                              # CY2023Q2 filed 2023-08-01 val=110
    trends = get_quarterly_trends(CIK, asof=pd.Timestamp("2023-09-01", tz="UTC"), client=_stub(facts))
    # asof 2023-09-01 sees the ORIGINAL Q1 (100, not the 2024 restatement) and Q2 (110)
    assert trends["revenue_yoy_prev"] == ...  # derived from 100/110, NOT 130

def test_asof_none_is_byte_identical_to_frame_path():
    a = get_quarterly_trends(CIK, client=_stub(facts))             # old path
    b = get_quarterly_trends(CIK, asof=None, client=_stub(facts))  # explicit None
    assert a == b

def test_accn_dedup_prevents_10k_10q_double_count():
    # same period reported in a 10-Q and the annual 10-K -> one value, latest filed<=asof
    ...
```

- [ ] **Step 2:** Run → FAIL (no `asof` param).
- [ ] **Step 3: Implement the as-of `units` path.** Add `asof` param. When set: iterate `units[unit]`, keep entries with `start`/`end` and a quarterly duration (`80 <= (end-start).days <= 100`), map to `(fiscal_year, quarter)` via `end`, drop entries with `filed > asof`, group by period and keep the row with max `filed` (dedup by `accn`), then feed the resulting per-period series into the existing `_trends_from_facts` math. When `asof is None`, call the unchanged frame-based path. Cache path: `…/<cik>/{(asof or now):%Y-%m-%d}.parquet`.
- [ ] **Step 4:** Run `tests/test_fundamentals_edgar.py -q` → PASS.
- [ ] **Step 5: Gate + commit** — `git commit -m "feat(edgar): as-of point-in-time get_quarterly_trends (audit #88)"`.

### Task 6: Thread `asof` from the pipeline + flip `fundamentals-restated` for covered names

**Files:**
- Modify: `tradinglib/scanner/pipeline.py:136` (pass `asof=`), and the per-name provenance from Task 4
- Test: `tests/test_scanner_pipeline*.py`

**Interfaces:**
- Consumes: Task 5's `asof` param; Task 4's per-name provenance
- Produces: when the scan runs with a historical `asof`, `get_quarterly_trends(int(cik), asof=asof, ...)`; names whose FA gate used EDGAR-PIT values omit `FUNDAMENTALS_RESTATED` from their `reasons`. Live (`asof=None`) behavior unchanged.

- [ ] **Step 1: Failing test** — a historical-asof scan: EDGAR-covered name's record `reasons` does NOT contain `FUNDAMENTALS_RESTATED` (but still has `MEMBERSHIP_SURVIVORSHIP`); a non-EDGAR name keeps both.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Pass `asof` to `get_quarterly_trends` at `pipeline.py:136`; compute per-name provenance as `merge(membership_prov, fundamentals_prov_if_not_edgar_pit)`.
- [ ] **Step 4:** Run → PASS; confirm live (`asof=None`) path byte-identical to Stage 2.
- [ ] **Step 5: Gate + commit** — `git commit -m "feat(pipeline): as-of EDGAR fundamentals + drop restated flag for covered names"`.

---

## Stage 4 — PIT prices (drops `price-restated`; supersedes A5b's lossy drop)

### Task 7: `load_daily` persists unadjusted OHLC + factors

**Files:**
- Modify: `tradinglib/loaders/equities/yfinance.py` (`_download_daily`, `_canonicalize`, `load_daily`)
- Test: `tests/test_*yfinance*` (find the equities loader test)

**Interfaces:**
- Produces: `load_daily(...)` frame keeps the existing adjusted `open/high/low/close/volume` columns UNCHANGED, plus new `unadj_open/unadj_high/unadj_low/unadj_close` columns (from `auto_adjust=False`). On cache miss, fetch `auto_adjust=False` and derive both.

- [ ] **Step 1: Failing test** — across a known split, the adjusted `close` series is back-adjusted (continuous) while `unadj_close` shows the raw pre-split price; both columns present; the adjusted columns equal what the old loader returned (byte-identical for indicators).
- [ ] **Step 2:** Run → FAIL (no `unadj_*`).
- [ ] **Step 3:** Change `_download_daily` to `auto_adjust=False`; in `_canonicalize` keep adjusted columns (computed from raw + Adj Close) AND add `unadj_*`. Heal old caches on read (recompute adjusted from raw if `unadj_*` absent → no `unadj_*` → flag downstream).
- [ ] **Step 4:** Run → PASS; an indicator (e.g. `sma`) computed on the new adjusted `close` equals the old value.
- [ ] **Step 5: Gate + commit** — `git commit -m "feat(prices): persist unadjusted OHLC alongside adjusted (audit #88)"`.

### Task 8: Ledger scores fills against unadjusted bars; flip `price-restated`

**Files:**
- Modify: `tradinglib/scanner/ledger.py` (score `simulate_ticket` against `unadj_*` when present), and the per-record provenance
- Test: `tests/test_scanner_ledger.py`

**Interfaces:**
- Consumes: Task 7's `unadj_*` columns; the A5b `corp_actions_since` path as fallback
- Produces: a ticket whose holding window crossed a split is scored against unadjusted bars (frozen dollar levels score correctly); its record omits `PRICE_RESTATED`. When `unadj_*` is unavailable, fall back to A5b's exclusion + keep `PRICE_RESTATED`.

- [ ] **Step 1: Failing test** — a ticket with a 2:1 split mid-hold: under the OLD adjusted scoring its dollar levels mis-score (target appears unreached); scored against `unadj_*` it resolves correctly (e.g. target hit). Assert the corrected R and that `PRICE_RESTATED` is absent; assert A5b fallback still fires when `unadj_*` missing.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** In the ledger, when the bars carry `unadj_*`, pass the unadjusted OHLC to `simulate_ticket`; drop `PRICE_RESTATED` from that record's reasons; else keep the A5b exclusion path + `PRICE_RESTATED`.
- [ ] **Step 4:** Run `tests/test_scanner_ledger.py -q` → PASS (incl. existing A5b tests).
- [ ] **Step 5: Gate + commit** — `git commit -m "feat(ledger): score against unadjusted bars; supersede A5b drop (audit #88)"`.

---

## Stage 5 — Parity + closeout

### Task 9: Before/after parity guarantee

**Files:**
- Create: `scripts/b_series_parity.py` (or extend `a2_resting_survival.py`'s pattern)
- Test: a parity test asserting `asof=None` + adjusted-path scanner output is byte-identical to `main` on a fixed fixture.

- [ ] **Step 1:** Write a parity check: run the scanner on a fixed synthetic/cached panel with the B branch; assert every numeric verdict/R/hit-rate equals the pre-B value (the only diffs are the added `leak`/`reasons` keys + the banner text). The corp-action-ticket scoring change (Task 8) is asserted SEPARATELY in `test_scanner_ledger.py`, not here.
- [ ] **Step 2:** Run → confirm zero numeric diffs.
- [ ] **Step 3: Commit** — `git commit -m "test(b-series): before/after parity guarantee"`.

### Task 10: Adversarial review + PR

- [ ] Run the full gate one final time.
- [ ] Adversarial multi-agent review of the diff (as for A2): verify the additive/zero-change guarantee, the as-of correctness (no future/restatement leak), unadjusted-scoring correctness, and that the banner/caveat fire exactly when leaked.
- [ ] Fold in confirmed findings; open the PR against `main` closing #88/#89/#90.
