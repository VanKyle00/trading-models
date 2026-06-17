"""Tests for the EDGAR companyfacts quarterly-trends loader."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from tradinglib.loaders.edgar_client import EdgarClient

_CIK = 320193


def _facts_with_filings() -> dict:
    """companyfacts with per-filing dates: an original Q1-2025 print, a LATER
    restatement of it, and a FUTURE Q1-2026 print — to exercise as-of selection."""

    def q(start: str, end: str, val: float, filed: str, accn: str, form: str = "10-Q") -> dict:
        return {"start": start, "end": end, "val": val, "filed": filed, "accn": accn, "form": form}

    return {
        "cik": _CIK,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            q("2024-01-01", "2024-03-31", 100e9, "2024-05-01", "a"),  # Q1-24 base
                            q(
                                "2025-01-01", "2025-03-31", 120e9, "2025-05-01", "b"
                            ),  # Q1-25 original
                            q(
                                "2025-01-01", "2025-03-31", 130e9, "2026-02-01", "b2", "10-K/A"
                            ),  # restated
                            q(
                                "2026-01-01", "2026-03-31", 150e9, "2026-05-01", "c"
                            ),  # Q1-26 (future)
                            q(
                                "2025-01-01", "2025-12-31", 500e9, "2026-02-01", "ann", "10-K"
                            ),  # annual
                        ]
                    }
                }
            }
        },
    }


def test_asof_uses_value_known_on_that_date_not_future_or_restatement() -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    # 2025-09-01: Q1-2026 not yet filed; the Q1-2025 restatement (filed 2026) not yet public.
    t = loader._trends_from_facts(_facts_with_filings(), asof=pd.Timestamp("2025-09-01", tz="UTC"))
    assert t["revenue_yoy"] == pytest.approx(120 / 100 - 1)  # ORIGINAL Q1-2025 over Q1-2024


def test_asof_later_sees_the_restatement_that_was_filed_by_then() -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    # 2026-03-01: the restatement (filed 2026-02) is public; Q1-2026 (filed 2026-05) still isn't.
    t = loader._trends_from_facts(_facts_with_filings(), asof=pd.Timestamp("2026-03-01", tz="UTC"))
    assert t["revenue_yoy"] == pytest.approx(130 / 100 - 1)  # restated Q1-2025 (as-of-known)


def test_asof_get_quarterly_trends_caches_by_asof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps(_facts_with_filings()))

    client = EdgarClient(
        transport=httpx.MockTransport(handler), clock=lambda: 0.0, sleep=lambda s: None
    )
    asof = pd.Timestamp("2025-09-01", tz="UTC")
    t = loader.get_quarterly_trends(_CIK, asof=asof, client=client)
    assert t["revenue_yoy"] == pytest.approx(0.20)
    # immutable as-of facts -> cache keyed by the asof date, not today
    assert (tmp_path / "fundamentals" / "edgar" / str(_CIK) / "2025-09-01.parquet").exists()


def _facts() -> dict:
    """Minimal companyfacts shape: quarterly revenue + diluted EPS frames."""

    def usd(frame: str, end: str, val: float) -> dict:
        return {"end": end, "val": val, "form": "10-Q", "frame": frame}

    return {
        "cik": _CIK,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            usd("CY2024Q1", "2024-03-30", 100e9),
                            usd("CY2024Q4", "2024-12-28", 110e9),
                            usd("CY2025Q1", "2025-03-29", 120e9),
                            usd("CY2025Q4", "2025-12-27", 121e9),
                            usd("CY2026Q1", "2026-03-28", 150e9),
                            # annual + instant frames must be ignored
                            {"end": "2025-09-27", "val": 400e9, "form": "10-K", "frame": "CY2025"},
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            usd("CY2025Q1", "2025-03-29", 1.0),
                            usd("CY2026Q1", "2026-03-28", 1.5),
                        ]
                    }
                },
            }
        },
    }


def test_trends_from_facts_yoy_and_acceleration() -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    trends = loader._trends_from_facts(_facts())

    assert trends["revenue_yoy"] == pytest.approx(150 / 120 - 1)  # latest quarter YoY
    assert trends["revenue_yoy_prev"] == pytest.approx(121 / 110 - 1)  # prior quarter YoY
    assert trends["revenue_accel"] == pytest.approx((150 / 120 - 1) - (121 / 110 - 1))
    assert trends["eps_change_yoy"] == pytest.approx(0.5)


def test_trends_missing_year_ago_quarter_is_nan() -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    facts = _facts()
    # drop the CY2025Q1 revenue point -> latest YoY has no base
    usd = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = [
        e for e in usd if e.get("frame") != "CY2025Q1"
    ]

    trends = loader._trends_from_facts(facts)

    assert np.isnan(trends["revenue_yoy"])


def test_trends_missing_tags_all_nan() -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    trends = loader._trends_from_facts({"facts": {"us-gaap": {}}})

    assert np.isnan(trends["revenue_yoy"])
    assert np.isnan(trends["revenue_accel"])
    assert np.isnan(trends["eps_change_yoy"])


def test_revenue_tag_fallback() -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    facts = _facts()
    gaap = facts["facts"]["us-gaap"]
    # primary tag name differs across filers; the loader tries alternatives
    gaap["RevenueFromContractWithCustomerExcludingAssessedTax"] = gaap.pop("Revenues")

    trends = loader._trends_from_facts(facts)

    assert trends["revenue_yoy"] == pytest.approx(150 / 120 - 1)


def test_get_quarterly_trends_fetches_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.fundamentals import edgar as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert str(request.url).endswith("CIK0000320193.json")
        return httpx.Response(200, text=json.dumps(_facts()))

    client = EdgarClient(
        transport=httpx.MockTransport(handler), clock=lambda: 0.0, sleep=lambda s: None
    )

    first = loader.get_quarterly_trends(_CIK, client=client)
    second = loader.get_quarterly_trends(_CIK, client=client)

    assert calls["n"] == 1  # second call served from the daily snapshot cache
    assert first == second
    assert first["revenue_yoy"] == pytest.approx(0.25)
