"""Tests for the scan pipeline orchestrator (all loaders mocked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.scanner.config import ScanConfig


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["DRIFT", "QUIET", "BROKEN"],
            "name": ["Drift Inc", "Quiet Corp", "Broken plc"],
            "sector": ["Tech", "Tech", "Tech"],
            "sub_industry": ["Software", "Software", "Software"],
            "cik": [1, 2, 3],
        }
    )


def _fundamentals(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": t,
                "snapshot": "2026-06-09",
                "market_cap": 1e11,
                "total_revenue": 1e10,
                "revenue_growth": 0.10,
                "earnings_growth": 0.10,
                "operating_margin": 0.20,
                "debt_to_equity": 100.0,
                "free_cashflow": 5e9,
                "forward_pe": 20.0,
                "trailing_pe": 25.0,
                "roe": 0.25,
                "avg_volume": 1e7,
            }
            for t in tickers
        ]
    )


def _pead_bars(symbol: str) -> pd.DataFrame:
    close = np.concatenate([np.full(110, 100.0), [108.0], np.linspace(108.2, 110.0, 9)])
    volume = np.concatenate([np.full(110, 1_000_000.0), [3_000_000.0], np.full(9, 1_200_000.0)])
    low = np.concatenate([np.full(110, 99.0), [105.0], np.linspace(107.0, 108.8, 9)])
    idx = pd.date_range("2026-01-01", periods=len(close), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
        },
        index=idx,
    )


def _quiet_bars(symbol: str) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(300) / 5.0)
    idx = pd.date_range("2025-01-01", periods=300, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(300, 1_000_000.0),
            "symbol": symbol,
        },
        index=idx,
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch):
    from tradinglib.scanner import pipeline

    bars_by_symbol = {
        "DRIFT": _pead_bars("DRIFT"),
        "QUIET": _quiet_bars("QUIET"),
        "SPY": _quiet_bars("SPY"),
    }

    def fake_load_daily(symbol, start=None, end=None, *, refresh=False):
        if symbol == "BROKEN":
            raise ValueError("yfinance returned no data for 'BROKEN'")
        return bars_by_symbol[symbol]

    drift_earnings = _pead_bars("DRIFT").index[110]
    next_earnings = _pead_bars("DRIFT").index[-1] + pd.Timedelta(days=4)  # shortly after asof

    def fake_earnings(tickers, start=None, end=None, *, refresh=False):
        rows = []
        for t in tickers:
            if t == "DRIFT":
                rows.append({"ticker": t, "earnings_datetime": drift_earnings, "session": "amc"})
                rows.append({"ticker": t, "earnings_datetime": next_earnings, "session": "amc"})
        return pd.DataFrame(rows, columns=["ticker", "earnings_datetime", "session"])

    monkeypatch.setattr(pipeline, "get_sp500_constituents", lambda *, refresh=False: _universe())
    monkeypatch.setattr(
        pipeline,
        "get_fundamental_snapshot",
        lambda tickers, *, refresh=False, max_workers=8: _fundamentals(tickers),
    )
    monkeypatch.setattr(pipeline, "load_daily", fake_load_daily)
    monkeypatch.setattr(pipeline, "get_earnings_dates", fake_earnings)
    monkeypatch.setattr(
        pipeline, "_now", lambda: pd.Timestamp(_pead_bars("x").index[-1]) + pd.Timedelta(days=1)
    )
    return pipeline


def test_run_scan_finds_setup_and_isolates_errors(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))

    assert result["funnel"]["universe"] == 3
    assert result["funnel"]["fa_shortlist"] == 3
    assert result["funnel"]["with_setups"] == 1

    tickers = [c["ticker"] for c in result["candidates"]]
    assert tickers == ["DRIFT"]
    assert result["candidates"][0]["setups"][0]["setup_type"] == "pead"

    # BROKEN failed in the bars stage but the scan completed
    assert len(result["errors"]) == 1
    assert result["errors"][0]["ticker"] == "BROKEN"


def test_run_scan_flags_upcoming_earnings(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))

    assert result["candidates"][0]["earnings_warning"] is True


def test_run_scan_respects_limit(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, limit=1, skip_llm=True))

    assert result["funnel"]["universe"] == 1


def test_run_scan_briefs_candidates_when_provider_given(
    patched_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def fake_briefs(provider, candidates, *, ciks, errors, refresh=False):
        seen["ciks"] = ciks
        for c in candidates:
            c["qualitative_score"] = 9.0
            c["stance"] = "favorable"
            c["brief"] = {"thesis": "good"}

    monkeypatch.setattr(patched_pipeline, "brief_candidates", fake_briefs)

    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3), provider=object())

    assert result["candidates"][0]["qualitative_score"] == 9.0
    assert result["candidates"][0]["brief"]["thesis"] == "good"
    assert seen["ciks"]["DRIFT"] == 1  # cik map comes from the universe


def test_run_scan_skips_briefs_with_skip_llm(
    patched_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"n": 0}
    monkeypatch.setattr(
        patched_pipeline, "brief_candidates", lambda *a, **k: called.update(n=called["n"] + 1)
    )

    patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True), provider=object())
    patched_pipeline.run_scan(ScanConfig(fa_keep=3), provider=None)

    assert called["n"] == 0
