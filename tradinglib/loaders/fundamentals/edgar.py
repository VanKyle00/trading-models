"""Quarterly fundamental trends from SEC EDGAR companyfacts (XBRL).

Extracts point-in-time quarterly revenue and diluted-EPS values from the
official ``data.sec.gov/api/xbrl/companyfacts`` API and derives trend
metrics the yfinance ``.info`` snapshot cannot provide:

- ``revenue_yoy``       — latest reported quarter vs the same quarter a year ago
- ``revenue_yoy_prev``  — the prior quarter's YoY (for acceleration)
- ``revenue_accel``     — ``revenue_yoy - revenue_yoy_prev``
- ``eps_change_yoy``    — diluted EPS change vs the year-ago quarter (additive,
  so it stays meaningful through zero/negative EPS)

Only entries with a quarterly calendar frame (``CY2026Q1``) are used — the
frames are the SEC's own deduplicated quarterly series, which sidesteps
10-K/amendment double-counting. The derived row (not the multi-MB raw JSON)
is snapshot-cached per day.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from tradinglib.data.paths import processed_dir
from tradinglib.loaders.edgar_client import EdgarClient

SOURCE = "fundamentals"
_SUBDIR = "edgar"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_QUARTER_FRAME = re.compile(r"^CY(\d{4})Q(\d)$")

_REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
_EPS_TAG = "EarningsPerShareDiluted"

_NAN = float("nan")

_default_client: EdgarClient | None = None


def _get_client(client: EdgarClient | None) -> EdgarClient:
    global _default_client
    if client is not None:
        return client
    if _default_client is None:
        _default_client = EdgarClient()
    return _default_client


def _quarterly_values(
    facts: dict, tags: tuple[str, ...], unit: str
) -> dict[tuple[int, int], float]:
    """``{(year, quarter): value}`` from the first tag that has quarterly frames."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        entries = gaap.get(tag, {}).get("units", {}).get(unit, [])
        values: dict[tuple[int, int], float] = {}
        for entry in entries:
            match = _QUARTER_FRAME.match(entry.get("frame") or "")
            if match:
                values[(int(match.group(1)), int(match.group(2)))] = float(entry["val"])
        if values:
            return values
    return {}


def _naive(ts: object) -> pd.Timestamp:
    """A tz-naive UTC timestamp (SEC ``filed``/``end`` are plain dates)."""
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _quarterly_values_asof(
    facts: dict, tags: tuple[str, ...], unit: str, asof: object
) -> dict[tuple[int, int], float]:
    """Point-in-time ``{(year, quarter): value}`` as publicly known on ``asof``.

    Reads the RAW ``units`` entries (not the SEC's hindsight-chosen ``frame``
    series, which carries no ``filed`` date): keeps quarterly-duration facts
    (~80-100 days), maps each to its calendar quarter by ``end``, drops anything
    ``filed`` after ``asof``, and per period keeps the LATEST filing on/before
    ``asof`` (so a restatement counts once it was public, never before). Annual
    figures fall out via the duration filter.
    """
    asof_n = _naive(asof)
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        entries = gaap.get(tag, {}).get("units", {}).get(unit, [])
        chosen: dict[tuple[int, int], tuple[pd.Timestamp, float]] = {}
        for entry in entries:
            start, end, filed = entry.get("start"), entry.get("end"), entry.get("filed")
            if not (start and end and filed):
                continue
            s, e, f = _naive(start), _naive(end), _naive(filed)
            if not (80 <= (e - s).days <= 100):  # quarterly durations only
                continue
            if f > asof_n:  # not yet filed as of asof
                continue
            # Key by the quarter's MIDPOINT, not its end: a 52/53-week filer's
            # quarter-end drifts a few days year over year and can cross a calendar
            # boundary (a winter quarter ending Dec-28 one year, Jan-2 the next),
            # which would scatter the same fiscal quarter across (Y,4) and (Y+1,1)
            # and break the year-ago lookup. The midpoint of a ~91-day quarter is
            # stable against that drift and matches the SEC frame's snapping.
            mid = s + (e - s) / 2
            period = (mid.year, (mid.month - 1) // 3 + 1)
            prev = chosen.get(period)
            if prev is None or f > prev[0]:  # latest filing on/before asof wins
                chosen[period] = (f, float(entry["val"]))
        if chosen:
            return {period: val for period, (_filed, val) in chosen.items()}
    return {}


def _year_ago(quarter: tuple[int, int]) -> tuple[int, int]:
    return (quarter[0] - 1, quarter[1])


def _prev_quarter(quarter: tuple[int, int]) -> tuple[int, int]:
    year, q = quarter
    return (year, q - 1) if q > 1 else (year - 1, 4)


def _yoy(values: dict[tuple[int, int], float], quarter: tuple[int, int]) -> float:
    base = values.get(_year_ago(quarter))
    current = values.get(quarter)
    if current is None or base is None or base == 0:
        return _NAN
    return current / base - 1.0


def _trends_from_facts(facts: dict, asof: object | None = None) -> dict[str, float]:
    if asof is None:
        revenue = _quarterly_values(facts, _REVENUE_TAGS, "USD")
        eps = _quarterly_values(facts, (_EPS_TAG,), "USD/shares")
    else:
        revenue = _quarterly_values_asof(facts, _REVENUE_TAGS, "USD", asof)
        eps = _quarterly_values_asof(facts, (_EPS_TAG,), "USD/shares", asof)

    revenue_yoy = revenue_yoy_prev = _NAN
    if revenue:
        latest = max(revenue)
        revenue_yoy = _yoy(revenue, latest)
        revenue_yoy_prev = _yoy(revenue, _prev_quarter(latest))

    eps_change = _NAN
    if eps:
        latest = max(eps)
        year_ago = eps.get(_year_ago(latest))
        if year_ago is not None:
            eps_change = eps[latest] - year_ago

    return {
        "revenue_yoy": revenue_yoy,
        "revenue_yoy_prev": revenue_yoy_prev,
        "revenue_accel": revenue_yoy - revenue_yoy_prev,
        "eps_change_yoy": eps_change,
    }


def get_quarterly_trends(
    cik: int,
    *,
    asof: object | None = None,
    refresh: bool = False,
    client: EdgarClient | None = None,
) -> dict[str, float]:
    """Derived quarterly-trend metrics for one company.

    ``asof=None`` (the default) uses the SEC's deduplicated quarterly frames and
    caches per day — today's behavior, byte-identical (legitimately PIT-equivalent
    at run time for a live scan). Pass an ``asof`` timestamp for a point-in-time
    historical replay: values reflect only filings public on/before that date, and
    the cache is keyed by ``asof`` (immutable, so it can persist indefinitely).
    """
    # As-of caches are namespaced ("asof-") apart from the asof=None daily cache:
    # the same calendar date means different things on the two paths (units as-of
    # vs frame today), so they must never share a parquet file.
    if asof is not None:
        fname = f"asof-{_naive(asof).strftime('%Y-%m-%d')}.parquet"
    else:
        fname = f"{pd.Timestamp.now('UTC').strftime('%Y-%m-%d')}.parquet"
    out = processed_dir(SOURCE) / _SUBDIR / str(cik) / fname
    if out.exists() and not refresh:
        return pd.read_parquet(out).iloc[0].to_dict()

    facts = json.loads(_get_client(client).get(_FACTS_URL.format(cik=cik)).text)
    trends = _trends_from_facts(facts, asof=asof)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([trends]).to_parquet(out)
    return trends
