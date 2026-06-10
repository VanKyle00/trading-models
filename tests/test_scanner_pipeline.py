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


def _fake_tournament_result(stance: str):
    from tradinglib.tournament.levels import Levels
    from tradinglib.tournament.run import StrategyVerdict, TournamentResult

    verdict = StrategyVerdict(
        key="sma_cross",
        survived=True,
        reasons=[],
        params={"fast": 10, "slow": 50},
        metrics={"deflated_sharpe": 0.95, "sharpe": 1.2},
        n_trades=20,
        param_stability={"fast": 0.0, "slow": 0.0},
        n_windows=6,
    )
    return TournamentResult(
        stance=stance,
        verdicts=[verdict],
        winner=verdict,
        winner_levels=Levels(
            entry=100.0, entry_type="market", stop=96.0, target=108.0, condition="test"
        ),
        n_trials=18,
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

    trend_calls = {"n": 0}

    def fake_trends(cik, *, refresh=False, client=None):
        trend_calls["n"] += 1
        return {
            "revenue_yoy": 0.10,
            "revenue_yoy_prev": 0.05,
            "revenue_accel": 0.05,
            "eps_change_yoy": 0.2,
        }

    monkeypatch.setattr(pipeline, "get_sp500_constituents", lambda *, refresh=False: _universe())
    monkeypatch.setattr(
        pipeline, "get_russell1000_constituents", lambda *, refresh=False: _universe()
    )
    monkeypatch.setattr(
        pipeline,
        "get_fundamental_snapshot",
        lambda tickers, *, refresh=False, max_workers=8: _fundamentals(tickers),
    )
    monkeypatch.setattr(pipeline, "load_daily", fake_load_daily)
    monkeypatch.setattr(pipeline, "get_earnings_dates", fake_earnings)
    monkeypatch.setattr(pipeline, "get_quarterly_trends", fake_trends)

    tournament_calls: list[tuple[str, str]] = []

    def fake_tournament(bars, stance, registry=None, config=None):
        tournament_calls.append((str(bars["symbol"].iloc[0]), stance))
        return _fake_tournament_result(stance)

    # raising=False: during this task's red phase the pipeline doesn't import
    # run_tournament yet; a strict setattr would error in fixture setup and
    # break the existing tests instead of letting the new ones fail cleanly.
    monkeypatch.setattr(pipeline, "run_tournament", fake_tournament, raising=False)
    pipeline._test_tournament_calls = tournament_calls

    pipeline._test_trend_calls = trend_calls  # for assertions
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


def test_run_scan_edgar_enrichment_runs_by_default(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))

    # one companyfacts fetch per pass-1 survivor (all 3 fixture tickers)
    assert patched_pipeline._test_trend_calls["n"] == 3
    assert result["funnel"]["fa_shortlist"] == 3


def test_run_scan_skip_edgar_flag(patched_pipeline) -> None:
    patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True, edgar_enrich=False))

    assert patched_pipeline._test_trend_calls["n"] == 0


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


def test_run_scan_emits_fa_candidates(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    longs = result["fa_candidates"]["long"]
    shorts = result["fa_candidates"]["short"]
    assert len(longs) == 1 and len(shorts) == 1
    assert longs[0]["rank"] == 1 and shorts[0]["rank"] == 1
    assert {"ticker", "name", "sector", "fa_score", "rank"} <= set(longs[0])
    assert longs[0]["ticker"] != shorts[0]["ticker"]
    assert result["funnel"]["fa_shortlist_short"] == 1
    assert result["config"]["universe"] == "russell1000"
    assert result["config"]["short_keep"] == 1


def test_run_scan_selects_universe_loader(patched_pipeline, monkeypatch) -> None:
    calls = {"sp500": 0, "russell1000": 0}

    def fake_sp500(*, refresh=False):
        calls["sp500"] += 1
        return _universe()

    def fake_r1000(*, refresh=False):
        calls["russell1000"] += 1
        return _universe()

    monkeypatch.setattr(patched_pipeline, "get_sp500_constituents", fake_sp500)
    monkeypatch.setattr(patched_pipeline, "get_russell1000_constituents", fake_r1000)

    patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))  # default universe
    patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True, universe="sp500"))

    assert calls == {"sp500": 1, "russell1000": 1}


def test_run_scan_skips_edgar_for_missing_cik(patched_pipeline, monkeypatch) -> None:
    uni = _universe()
    uni["cik"] = pd.array([1, pd.NA, 3], dtype="Int64")  # QUIET loses its CIK
    monkeypatch.setattr(
        patched_pipeline, "get_russell1000_constituents", lambda *, refresh=False: uni
    )

    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))

    edgar_errors = [e for e in result["errors"] if e["stage"] == "edgar"]
    assert any(e["ticker"] == "QUIET" and "CIK" in e["error"] for e in edgar_errors)
    assert patched_pipeline._test_trend_calls["n"] == 2  # the other two still fetched


def test_run_scan_reports_stale_universe(patched_pipeline, monkeypatch) -> None:
    uni = _universe()
    uni.attrs["snapshot"] = "2026-06-01"  # loader served an old snapshot
    monkeypatch.setattr(
        patched_pipeline, "get_russell1000_constituents", lambda *, refresh=False: uni
    )

    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))

    stale = [e for e in result["errors"] if e["stage"] == "universe"]
    assert len(stale) == 1
    assert "2026-06-01" in stale[0]["error"]


def test_run_scan_runs_tournament_on_fa_candidates(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    entries = result["tournament"]["long"] + result["tournament"]["short"]
    t_errors = [e for e in result["errors"] if e["stage"] == "tournament"]
    # one long + one short candidate; each either produced an entry or an error
    assert len(entries) + len(t_errors) == 2
    assert entries  # at most one fixture ticker (BROKEN) can fail bar loading
    for entry in entries:
        assert entry["winner"]["strategy"] == "sma_cross"
        assert entry["winner"]["levels"]["entry"] == 100.0
        assert entry["survivors"] == ["sma_cross"]
        assert entry["winner_changed"] is None  # no previous report
        assert entry["n_trials"] == 18
        assert entry["verdicts"][0]["survived"] is True
    assert result["funnel"]["tournament_candidates"] == 2
    assert result["funnel"]["tournament_survivors"] == len(entries)
    assert result["config"]["skip_strategies"] is False


def test_run_scan_skip_strategies(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(
        ScanConfig(fa_keep=1, short_keep=1, skip_llm=True, skip_strategies=True)
    )

    assert patched_pipeline._test_tournament_calls == []
    assert result["tournament"] == {"long": [], "short": []}
    assert result["funnel"]["tournament_candidates"] == 0
    assert result["funnel"]["tournament_survivors"] == 0
    assert result["config"]["skip_strategies"] is True


def test_run_scan_winner_changed_against_previous_report(patched_pipeline) -> None:
    config = ScanConfig(fa_keep=1, short_keep=1, skip_llm=True)
    first = patched_pipeline.run_scan(config)

    unchanged = patched_pipeline.run_scan(config, previous_report=first)
    entries = unchanged["tournament"]["long"] + unchanged["tournament"]["short"]
    assert entries and all(e["winner_changed"] is False for e in entries)

    for stance in ("long", "short"):
        for e in first["tournament"][stance]:
            e["winner"] = None  # pretend last night had no survivor here
    flipped = patched_pipeline.run_scan(config, previous_report=first)
    entries = flipped["tournament"]["long"] + flipped["tournament"]["short"]
    assert entries and all(e["winner_changed"] is True for e in entries)


def test_run_scan_isolates_tournament_failures(patched_pipeline, monkeypatch) -> None:
    def exploding(bars, stance, registry=None, config=None):
        raise RuntimeError("walk-forward blew up")

    monkeypatch.setattr(patched_pipeline, "run_tournament", exploding)
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    t_errors = [e for e in result["errors"] if e["stage"] == "tournament"]
    assert len(t_errors) == 2  # both candidates failed, scan completed
    assert result["tournament"] == {"long": [], "short": []}
    assert result["funnel"]["tournament_candidates"] == 2
    assert result["funnel"]["tournament_survivors"] == 0
    assert result["fa_candidates"]["long"]  # earlier stages untouched


def test_run_scan_tournament_loads_longer_history(patched_pipeline, monkeypatch) -> None:
    starts: list[str] = []
    original = patched_pipeline.load_daily

    def recording(symbol, start=None, end=None, *, refresh=False):
        starts.append(start)
        return original(symbol, start=start, end=end, refresh=refresh)

    monkeypatch.setattr(patched_pipeline, "load_daily", recording)
    patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    asof = patched_pipeline._now()
    expected = (asof - pd.Timedelta(days=1200)).strftime("%Y-%m-%d")
    assert expected in starts  # tournament bars span ~3y+, not the 450d watchlist window
