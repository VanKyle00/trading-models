# B-series — Point-in-Time Honesty Layer (design)

**Date:** 2026-06-17
**Audit issues closed:** #88 (no PIT data), #89 (survivorship not machine-enforced), #90 (pooled-cert promotes on survivor-only evidence)
**Out of scope / kept separate:** A1 deferrals (#93), real point-in-time *membership* (no free data source exists)

## Context & trust boundary

The pipeline (scanner → tournament → live tickets + forward ledger, plus the no-lookahead
`backfill_scan` replay) is sound for **relative A/B deltas** but **inflates absolute R / hit-rate**,
because the data is not point-in-time (PIT) and is survivorship-biased. The audit's headline holds:
biases are common-mode and cancel in a delta; they do **not** cancel in absolute numbers.

The chosen remedy is a **hybrid**: deliver true PIT where free data (yfinance / EDGAR) supports it,
and machine-enforce a `leak=True` / `BIASED-UPPER-BOUND` honesty banner everywhere else.

### Non-goal (the honest ceiling)

B does **not** turn absolute numbers into a trustworthy edge. Index-membership survivorship is
**unfixable on free data** (see §3c), so absolute results will *always* carry a banner. B's job is to
(a) make every absolute number **honest** (machine-flagged), and (b) **remove the fixable leaks**
(fundamentals restatement, price restatement) so the flagged numbers are as clean as free data allows.
Relative A/B comparisons remain sound throughout and are unaffected.

## 1. The three leak axes (from the surface map)

| Axis | Real PIT on free data? | Treatment |
|---|---|---|
| **EDGAR fundamentals** (`get_quarterly_trends`) | **Yes, well-bounded.** SEC `units[]` carries per-fact `filed`/`end`/`accn`. | real PIT (`leak=False` arm) |
| **Prices** (`yfinance` `auto_adjust=True`) | **Yes.** `auto_adjust=False` gives unadjusted bars for free; the code discards them. | real PIT (`leak=False` arm) |
| **Index membership** (`russell1000`/`sp500`) | **No.** Both scrape *today's* Wikipedia table; no historical/as-of membership in any free source. | permanent banner (`leak=True`) |

## 2. Provenance model (the spine)

New module `tradinglib/provenance.py`, generalizing the convention that today lives only in
`scripts/backfill_scan.py` (`point_in_time_status`, `honesty_summary`, the `BIASED-UPPER-BOUND` banner):

```python
@dataclass(frozen=True)
class Provenance:
    leak: bool                  # True if any non-PIT / survivor axis is present
    reasons: tuple[str, ...]    # subset of {"membership-survivorship",
                                #            "fundamentals-restated",
                                #            "price-restated"}

def merge(*provs: Provenance) -> Provenance: ...      # union of reasons; leak = any
def honesty_summary(records: list[dict]) -> dict: ...  # {leaked, leaked_records, honest_records}
BIASED_UPPER_BOUND_BANNER: str                         # one canonical banner string
```

`backfill_scan.py` is refactored to import these (single source of truth; its behavior is unchanged —
`point_in_time_status` becomes a thin shim that returns the family-selection reason).

**Decision:** `leak: bool` + `reasons` tuple — simple, and the reasons tuple already names which axes
are dirty. No richer per-axis structure.

## 3. The three arms

### 3a. EDGAR-PIT fundamentals (`leak=False` arm)

`get_quarterly_trends(cik, *, asof: pd.Timestamp | None = None, refresh=False, client=None)` — same flat
4-key return shape (`revenue_yoy`, `revenue_yoy_prev`, `revenue_accel`, `eps_change_yoy`), so no consumer
column change. When `asof` is set:

- Read the raw `facts.us-gaap.<tag>.units.<unit>[]` array (carries `filed`/`start`/`end`/`accn`/`form`)
  instead of the `frame`-keyed series (which carries **no** `filed` date — the docstring's "point-in-time"
  claim is wrong; frames are de-duped, not as-of-filtered).
- Keep quarterly-duration facts (`end - start` ≈ 80–100 days), map each to its fiscal quarter via `end`,
  **dedup by `accn`**, and keep the latest fact with `filed <= asof` per `(year, quarter)`.
- Key the snapshot cache by `asof` (`…/<cik>/{asof or today:%Y-%m-%d}.parquet`) — as-of facts for past
  dates are immutable, so cache-forever (mirrors `get_filing_text`).

`asof=None` routes the **existing frame-based path entirely unchanged** (byte-identical); only
`asof != None` uses the new `units`-based as-of path. The **live nightly scan passes `asof=None`**
(today's fundamentals are legitimately PIT-equivalent at run time); the **backfill/replay passes the
night's date**, which is where true as-of matters. Thread `asof` from `pipeline.py:136`. The sibling
`filings/edgar.py:get_recent_filings` (asof param + `filing_date` + asof-keyed cache) is the template.

A historical name whose FA-gate decision rests on EDGAR-PIT values drops the `fundamentals-restated` reason.

### 3b. PIT prices (`leak=False` arm; supersedes A5b's lossy drop)

`load_daily` (`equities/yfinance.py`) fetches `auto_adjust=False` and persists **both** the unadjusted
OHLC **and** the split/dividend adjustment factors (from `Adj Close`/raw or `yf.Ticker().actions`). The
canonical **adjusted** columns are kept unchanged for indicator/feature/signal math (back-adjusted
continuity is correct there). New **unadjusted** OHLC columns are added.

The forward **ledger scores fills against the unadjusted bars** — the issued levels were dollar prices on
the issue-night tape, so this removes the `simulate_ticket` rescale mismatch and the restatement leak,
upgrading A5b's detect-and-**drop** (lossy) to score-**correctly** (keeps the sample). Drops the
`price-restated` reason for scored tickets.

A5b's `corp_actions_since` exclusion is **kept as a fallback** when unadjusted reconstruction is
unavailable (additive, no regression).

### 3c. Membership — permanent banner (`leak=True`)

No historical/as-of membership exists on free sources (Wikipedia current table only; Russell has no
history table). Every result whose ticker set derives from `russell1000`/`sp500` carries `leak=True` with
reason `membership-survivorship`, carried via the existing `df.attrs['snapshot']` (download date).
Two leak points to flag explicitly: `pipeline.py:108-115` only errors when `snapshot != today` (implicitly
asserting same-day membership is correct for *any* as-of date — it is not), and `backfill_scan`'s
"no-lookahead" replay slices bars/earnings cleanly but inherits a today-sourced ticker set (its banner must
not claim membership is PIT).

## 4. Data flow / threading

The leak/banner is about **historical / absolute evidence** — the backfill-replay prints, the ledger's
absolute aggregates, and the survivor-biased cross-sectional evidence that certifies a pooled ticket. It is
NOT a claim that a live forward ticket "leaks" (a forward ticket issued tonight on today's data is honestly
as-of-now). The banner answers "is this *absolute backtest/replay number* a biased upper bound", and the
pooled-cert caveat answers "was this promotion certified on survivor-biased historical evidence".

- **Origin** (`run_scan`, `pipeline.py:84+`): compute per-name `Provenance` for the historical/evidence
  framing. Default today = `leak=True` (`membership-survivorship` always, because replay/evidence back-projects
  today's members; `fundamentals-restated` until 3a's as-of path feeds the gate; `price-restated` until 3b
  lands). Structure it per-record so names flip reasons off as the PIT arms land.
- **Propagate**: add `leak`/`reasons` to the record dicts through `pipeline → ledger → pooled` (the
  pooled-promoted ticket, `pipeline.py:306-323`, carries the caveat — closes #90).
- **Surface**: `honesty_summary` drives the `BIASED-UPPER-BOUND` banner on every absolute-number print;
  the pooled-promoted ticket shows its provenance reasons.

## 5. Zero-change guarantee (verified like the A-series)

- `asof=None` ⇒ byte-identical fundamentals (3a is additive).
- Adjusted columns unchanged ⇒ all indicators / signals / tournament / DSR byte-identical (3b adds columns,
  doesn't alter the adjusted series indicators read).
- The `leak` flag is **metadata + banner only** — it never changes an R / hit-rate value, only annotates it.
  #90's caveat is additive.
- A before/after parity check (flag-off-equivalent vs new) confirms absolute numbers are unchanged in value;
  only the banner/metadata is added. The ledger-scoring change in 3b *does* change scored R for
  corp-action-affected tickets — that is the intended correctness fix, asserted by a dedicated test, not the
  parity check.

## 6. Testing (TDD)

- **provenance**: `Provenance.merge` union/leak logic; `honesty_summary` aggregation; banner fires iff leaked.
- **EDGAR as-of**: fixture with multiple filings + a later restatement for a past quarter → `asof` picks the
  latest `filed <= asof`, ignores future filings AND later restatements; `asof=None` == current behavior;
  `accn` dedup prevents 10-K/10-Q double-count.
- **PIT prices**: split fixture → unadjusted vs adjusted diverge; the ledger scores the frozen dollar levels
  against unadjusted bars (a winner that adjusted-rescaling would mislabel is scored correctly); adjusted
  columns unchanged ⇒ an indicator computed before/after is identical.
- **membership banner**: a result sourced from the universe loaders carries `leak=True`/`membership-survivorship`.
- **parity**: `asof=None` + adjusted path ⇒ byte-identical scanner output to current `main`.

## 7. Implementation phasing (one branch, staged commits)

1. `provenance.py` + refactor `backfill_scan` onto it (no behavior change).
2. Membership banner + leak threading through `pipeline → ledger → pooled` (default `leak=True`); #89/#90 close at the honesty level.
3. EDGAR-PIT fundamentals (`asof`) + flip `fundamentals-restated` off for EDGAR-covered names.
4. PIT prices (unadjusted + ledger scoring) + flip `price-restated` off; keep A5b fallback.

## 8. Risks & mitigations (from the map)

- **EDGAR raw-`units` dedup**: switching off `frame` re-introduces the 10-K/10-Q double-count the frame
  avoided → MUST dedup by `accn` and pick latest-`filed`-≤-`asof` per period.
- **Restatement leakage**: today's companyfacts embeds restated past-quarter numbers; the `filed <= asof`
  filter is what removes them — the core of the as-of correctness.
- **yfinance unadjusted reliability**: if `auto_adjust=False` reconstruction is unreliable for a symbol,
  fall back to A5b's corp-action exclusion + `leak`/`price-restated` flag rather than mis-scoring.
- **A5b overlap**: 3b touches the same ledger scoring as A5b; A5b's exclusion stays as the fallback path,
  so there is no regression if 3b's unadjusted scoring is unavailable.
