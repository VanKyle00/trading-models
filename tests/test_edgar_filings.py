"""Tests for the EDGAR filings loader (submissions index + filing text)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

_FIXTURES = Path(__file__).parent / "fixtures" / "edgar"
_CIK = 320193


def _client_returning(payload_by_suffix: dict[str, str]):
    from tradinglib.loaders.edgar_client import EdgarClient

    def handler(request: httpx.Request) -> httpx.Response:
        for suffix, payload in payload_by_suffix.items():
            if str(request.url).endswith(suffix):
                return httpx.Response(200, text=payload)
        return httpx.Response(404, text="missing")

    fake_sleep: list[float] = []
    return EdgarClient(
        transport=httpx.MockTransport(handler),
        clock=lambda: 0.0,
        sleep=fake_sleep.append,
    )


def test_get_recent_filings_filters_forms_and_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.filings import edgar as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    client = _client_returning({"CIK0000320193.json": (_FIXTURES / "submissions.json").read_text()})

    df = loader.get_recent_filings(
        _CIK, asof=pd.Timestamp("2026-06-09", tz="UTC"), days=120, client=client
    )

    # the January 8-K (130 days back) and the 2023 10-K fall outside the
    # 120-day window; the June 8-K and May 10-Q are inside
    assert list(df.columns) == ["cik", "form", "accession", "filing_date", "primary_document"]
    assert set(df["form"]) == {"8-K", "10-Q"}
    assert len(df) == 2
    # newest first
    assert df.iloc[0]["filing_date"] >= df.iloc[-1]["filing_date"]


def test_get_recent_filings_snapshot_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.filings import edgar as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    payload = (_FIXTURES / "submissions.json").read_text()

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text=payload)

    from tradinglib.loaders.edgar_client import EdgarClient

    client = EdgarClient(
        transport=httpx.MockTransport(handler), clock=lambda: 0.0, sleep=lambda s: None
    )

    loader.get_recent_filings(_CIK, asof=pd.Timestamp("2026-06-09", tz="UTC"), client=client)
    loader.get_recent_filings(_CIK, asof=pd.Timestamp("2026-06-09", tz="UTC"), client=client)

    assert calls["n"] == 1


def test_get_filing_text_strips_html_and_caches_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.filings import edgar as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    html = (_FIXTURES / "filing.html").read_text()
    client = _client_returning({"aapl-20260328.htm": html})

    text = loader.get_filing_text(_CIK, "0000320193-26-000040", "aapl-20260328.htm", client=client)

    assert "Net sales increased 8%" in text
    assert "should never appear" not in text  # scripts stripped
    assert "color: red" not in text  # styles stripped
    assert "Management" in text and "Discussion" in text

    # cached on disk keyed by accession; a second call needs no client at all
    again = loader.get_filing_text(
        _CIK, "0000320193-26-000040", "aapl-20260328.htm", client=_client_returning({})
    )
    assert again == text


def test_get_filing_text_respects_max_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.filings import edgar as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    html = (_FIXTURES / "filing.html").read_text()
    client = _client_returning({"aapl-20260328.htm": html})

    text = loader.get_filing_text(
        _CIK, "0000320193-26-000040", "aapl-20260328.htm", max_chars=40, client=client
    )

    assert len(text) <= 40


def test_submissions_json_shape_parses() -> None:
    # guard: the fixture matches the real data.sec.gov shape we parse
    payload = json.loads((_FIXTURES / "submissions.json").read_text())
    recent = payload["filings"]["recent"]
    assert {"accessionNumber", "form", "filingDate", "primaryDocument"} <= set(recent)
