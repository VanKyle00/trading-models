"""Tests for the EDGAR companyfacts quarterly-trends loader."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from tradinglib.loaders.edgar_client import EdgarClient

_CIK = 320193


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
