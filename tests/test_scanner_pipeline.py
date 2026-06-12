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
def patched_pipeline(monkeypatch: pytest.MonkeyPatch, make_chain):
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

    chain_mids = {
        ("call", 75, 95.0): 8.0,
        ("call", 75, 110.0): 2.4,
        ("put", 38, 90.0): 1.0,
        ("put", 38, 95.0): 2.8,
        ("put", 75, 105.0): 7.5,
        ("put", 75, 90.0): 1.8,
        ("call", 38, 105.0): 2.4,
        ("call", 38, 110.0): 0.7,
    }

    def fake_fetch_chain(ticker, *, max_days=120):
        return make_chain(mids=chain_mids)

    # raising=False for the same red-phase reason as run_tournament above
    monkeypatch.setattr(pipeline, "fetch_chain", fake_fetch_chain, raising=False)

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
    # BROKEN is the short FA candidate (after EDGAR) and fails bar loading in the
    # setup-detect loop before reaching the tournament, so it contributes neither
    # an entry nor a tournament-stage error; DRIFT (long) always produces an entry.
    assert len(entries) == 1  # only DRIFT (long) reaches the tournament
    bars_errors = [e for e in result["errors"] if e["stage"] == "bars"]
    assert len(bars_errors) == 1 and bars_errors[0]["ticker"] == "BROKEN"
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
    # BROKEN is the short FA candidate and fails bar loading in the setup-detect loop
    # (before the tournament), so only DRIFT (long) reaches run_tournament and errors.
    assert len(t_errors) == 1  # DRIFT failed in tournament; BROKEN failed bars earlier
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


def test_run_scan_tournament_drops_nan_ohlc_rows(patched_pipeline, monkeypatch) -> None:
    # yfinance sometimes appends a price-less row (volume only, OHLC all NaN)
    # for the most recent session; one such row inside the last ATR window
    # poisons the winner's levels and cost a real survivor in the live smoke.
    captured: list[pd.DataFrame] = []

    def capturing(bars, stance, registry=None, config=None):
        captured.append(bars)
        return _fake_tournament_result(stance)

    monkeypatch.setattr(patched_pipeline, "run_tournament", capturing)

    original = patched_pipeline.load_daily

    def with_nan_tail(symbol, start=None, end=None, *, refresh=False):
        bars = original(symbol, start=start, end=end, refresh=refresh).copy()
        bars.iloc[-1, bars.columns.get_indexer(["open", "high", "low", "close"])] = float("nan")
        return bars

    monkeypatch.setattr(patched_pipeline, "load_daily", with_nan_tail)
    patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    assert captured
    for bars in captured:
        assert not bars[["open", "high", "low", "close"]].isna().any().any()


def test_run_scan_builds_tickets_for_winners(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    entries = result["tournament"]["long"] + result["tournament"]["short"]
    with_winner = [e for e in entries if e["winner"]]
    tickets = result["tickets"]["long"] + result["tickets"]["short"]
    assert tickets and len(tickets) == len(with_winner)
    for t in tickets:
        assert sum(s["recommended"] for s in t["structures"]) == 1
        assert t["evidence"]["fa_rank"] == 1  # fa_keep=1 -> the sole candidate ranks 1
        assert t["evidence"]["deflated_sharpe"] == 0.95
        assert any("indicative" in w for w in t["warnings"])
    assert result["funnel"]["tickets"] == len(tickets)
    assert result["config"]["account_size"] == 100_000.0
    assert result["config"]["risk_per_trade_pct"] == 0.01


def test_run_scan_ticket_chain_failure_degrades_to_stock_only(
    patched_pipeline, monkeypatch
) -> None:
    def exploding_fetch(ticker, *, max_days=120):
        raise RuntimeError("yfinance chain flake")

    monkeypatch.setattr(patched_pipeline, "fetch_chain", exploding_fetch)
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    tickets = result["tickets"]["long"] + result["tickets"]["short"]
    assert tickets  # the stock plan still ships
    for t in tickets:
        assert [s["kind"] for s in t["structures"]] in (["stock"], ["stock_short"])
        assert any("options_illiquid" in w for w in t["warnings"])
    assert any(e["stage"] == "tickets" for e in result["errors"])


def test_run_scan_isolates_ticket_failures(patched_pipeline, monkeypatch) -> None:
    def exploding_build(**kwargs):
        raise RuntimeError("playbook blew up")

    monkeypatch.setattr(patched_pipeline, "build_ticket", exploding_build)
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    assert result["tickets"] == {"long": [], "short": []}
    t_errors = [e for e in result["errors"] if e["stage"] == "tickets"]
    assert t_errors  # every winner errored, scan completed
    assert result["tournament"]["long"] or result["tournament"]["short"]  # earlier stages intact
    assert result["funnel"]["tickets"] == 0


def test_run_scan_skip_strategies_means_no_tickets(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(
        ScanConfig(fa_keep=1, short_keep=1, skip_llm=True, skip_strategies=True)
    )

    assert result["tickets"] == {"long": [], "short": []}
    assert result["funnel"]["tickets"] == 0


def test_run_scan_tournament_bars_carry_earnings_column(patched_pipeline, monkeypatch) -> None:
    """Bars passed into run_tournament must have a bool 'earnings' column."""
    captured: list[pd.DataFrame] = []

    def capturing(bars, stance, registry=None, config=None):
        captured.append(bars.copy())
        return _fake_tournament_result(stance)

    monkeypatch.setattr(patched_pipeline, "run_tournament", capturing)
    patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    assert captured, "run_tournament was never called"
    for bars in captured:
        assert "earnings" in bars.columns, "earnings column missing"
        assert bars["earnings"].dtype == bool, f"dtype was {bars['earnings'].dtype}, expected bool"

    # The fixture stubs drift_earnings = index[110] for DRIFT; that position
    # must be True and the out-of-range next_earnings must NOT be flagged.
    drift_bars = next(b for b in captured if b["symbol"].iloc[0] == "DRIFT")
    assert bool(drift_bars["earnings"].iloc[110]) is True, "reaction bar not flagged"
    # next_earnings is after the last bar so no extra bar should be True beyond index 110
    flagged_positions = drift_bars["earnings"].to_numpy().nonzero()[0]
    assert list(flagged_positions) == [110], f"unexpected flagged positions: {flagged_positions}"


def test_run_scan_tournament_earnings_fetch_failure_attaches_all_false_column(
    patched_pipeline, monkeypatch
) -> None:
    """If get_earnings_dates raises in the tournament stage, the bars still get an
    all-False earnings column and an error entry is appended with stage='tournament'
    and 'earnings' in the message."""
    captured: list[pd.DataFrame] = []

    def capturing(bars, stance, registry=None, config=None):
        captured.append(bars.copy())
        return _fake_tournament_result(stance)

    monkeypatch.setattr(patched_pipeline, "run_tournament", capturing)

    call_count = {"n": 0}
    original_earnings = patched_pipeline.get_earnings_dates

    def earnings_that_fails_on_second_call(tickers, start=None, end=None, *, refresh=False):
        call_count["n"] += 1
        # First call is from the setups stage (bars loop); second+ are from tournament.
        # We want to fail the tournament-stage call(s) only.
        if call_count["n"] > 1:
            raise RuntimeError("earnings API unavailable")
        return original_earnings(tickers, start=start, end=end, refresh=refresh)

    monkeypatch.setattr(patched_pipeline, "get_earnings_dates", earnings_that_fails_on_second_call)

    result = patched_pipeline.run_scan(ScanConfig(fa_keep=1, short_keep=1, skip_llm=True))

    # Scan must not skip the ticker — run_tournament should still have been called.
    assert captured, "tournament was skipped entirely — ticker should not be dropped"

    for bars in captured:
        assert "earnings" in bars.columns, "earnings column missing on failure path"
        assert bars["earnings"].dtype == bool, f"dtype was {bars['earnings'].dtype}"
        assert not bars["earnings"].any(), "all-False expected when earnings fetch raises"

    t_errors = [e for e in result["errors"] if e["stage"] == "tournament"]
    assert t_errors, "no tournament-stage error recorded"
    assert any("earnings" in e["error"] for e in t_errors), (
        f"'earnings' not in error message: {t_errors}"
    )


# ---------------------------------------------------------------------------
# FDR tiering tests
#
# BH math for the stub family of 2 (fa_keep=1, short_keep=1):
#   DRIFT long:  DSR=0.999 → p=0.001  rank 1 → 0.001 <= (1/2)*0.10=0.050  PASS
#   QUIET short: DSR=0.92  → p=0.08   rank 2 → 0.08  <= (2/2)*0.10=0.100  PASS (0.08≤0.10)
#
# With m=2, QUIET (p=0.08) still passes (BH threshold 0.10 at rank 2).
# To make QUIET fail we need a larger family.  Patch apply_fdr in the
# pipeline module to return a controlled result that demotes QUIET — this
# tests the pipeline's routing logic; BH arithmetic is already covered by
# test_scanner_tiers.py::test_apply_fdr_passes_only_overwhelming_evidence.
#
# Concrete stub values used in fdr_pipeline:
#   - DRIFT long DSR=0.999 (p=0.001) is the "passing" entry
#   - QUIET long DSR=0.92  (p=0.08)  is the "failing" entry
#   - apply_fdr is patched to return: DRIFT→True, QUIET→False (threshold=0.001, family=2)
# ---------------------------------------------------------------------------


def _fake_fdr_tournament_result(stance: str, ticker: str):
    """Return DSR=0.999 for DRIFT, DSR=0.92 for QUIET."""
    from tradinglib.tournament.levels import Levels
    from tradinglib.tournament.run import StrategyVerdict, TournamentResult

    dsr = 0.999 if ticker == "DRIFT" else 0.92
    verdict = StrategyVerdict(
        key="sma_cross",
        survived=True,
        reasons=[],
        params={"fast": 10, "slow": 50},
        metrics={"deflated_sharpe": dsr, "sharpe": dsr + 0.2},
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


@pytest.fixture
def fdr_pipeline(patched_pipeline, monkeypatch):
    """patched_pipeline with controlled DSR + patched apply_fdr for routing tests.

    Stub: DRIFT long passes (DSR=0.999), QUIET long fails (DSR=0.92).
    apply_fdr is patched to return the controlled pass/fail map so pipeline
    routing is tested independently of BH arithmetic (covered by tiers tests).
    """
    from tradinglib.scanner import pipeline

    def fake_fdr_tournament(bars, stance, registry=None, config=None):
        ticker = str(bars["symbol"].iloc[0])
        return _fake_fdr_tournament_result(stance, ticker)

    monkeypatch.setattr(pipeline, "run_tournament", fake_fdr_tournament)

    # Patch apply_fdr: DRIFT long passes, QUIET long fails.
    # fa_keep=1 → only DRIFT is a long candidate; short_keep=1 → QUIET is the short.
    # We demote QUIET regardless of stance to verify watchlist routing.
    def fake_apply_fdr(tournament, alpha):
        passed = {}
        for stance in ("long", "short"):
            for entry in tournament.get(stance) or []:
                ticker = entry["ticker"]
                passed[(stance, ticker)] = ticker == "DRIFT"
        family = sum(1 for v in passed.values())
        threshold = 0.001 if any(v for v in passed.values()) else 0.0
        return passed, threshold, family

    monkeypatch.setattr(pipeline, "apply_fdr", fake_apply_fdr, raising=False)
    return pipeline


def test_fdr_tickets_only_for_passing_entries(fdr_pipeline) -> None:
    # fa_keep=2, short_keep=0: DRIFT and QUIET both enter as long candidates.
    # apply_fdr is patched: DRIFT→True, QUIET→False.
    # Only DRIFT should get a ticket; QUIET is demoted to watchlist.
    result = fdr_pipeline.run_scan(ScanConfig(fa_keep=2, short_keep=0, skip_llm=True))

    tickets = result["tickets"]["long"] + result["tickets"]["short"]
    tickers_with_tickets = {t["ticker"] for t in tickets}
    assert "DRIFT" in tickers_with_tickets, "DRIFT must be ticketed (passes FDR)"
    assert "QUIET" not in tickers_with_tickets, "QUIET must NOT be ticketed (fails FDR)"
    assert all(t["tier"] == "ticket" for t in tickets), "all tickets must carry tier='ticket'"


def test_fdr_demoted_entries_land_in_watchlist(fdr_pipeline) -> None:
    result = fdr_pipeline.run_scan(ScanConfig(fa_keep=2, short_keep=0, skip_llm=True))

    watchlist = result["watchlist"]
    wl_rows = watchlist.get("long", []) + watchlist.get("short", [])
    quiet_rows = [r for r in wl_rows if r["ticker"] == "QUIET"]
    assert quiet_rows, "QUIET must appear in watchlist after failing BH"
    for r in quiet_rows:
        assert r["tier_reason"] == "failed nightly FDR"


def test_fdr_report_keys(fdr_pipeline) -> None:
    result = fdr_pipeline.run_scan(ScanConfig(fa_keep=2, short_keep=0, skip_llm=True))

    assert result["fdr"] is not None
    assert result["fdr"]["alpha"] == 0.10
    assert isinstance(result["fdr"]["threshold"], float)
    assert isinstance(result["fdr"]["family"], int)


def test_fdr_funnel_keys(fdr_pipeline) -> None:
    result = fdr_pipeline.run_scan(ScanConfig(fa_keep=2, short_keep=0, skip_llm=True))

    assert "fdr_passed" in result["funnel"]
    assert "watchlist" in result["funnel"]
    # DRIFT long passes; QUIET long fails → 1 passed, 1 in watchlist
    assert result["funnel"]["fdr_passed"] == 1
    assert result["funnel"]["watchlist"] >= 1


def test_fdr_config_echo(fdr_pipeline) -> None:
    result = fdr_pipeline.run_scan(ScanConfig(fa_keep=2, short_keep=0, skip_llm=True))

    assert result["config"]["fdr_alpha"] == 0.10
    assert result["config"]["watch_dsr_floor"] == 0.5


def test_fdr_skip_strategies_zeroes_out(patched_pipeline) -> None:
    result = patched_pipeline.run_scan(
        ScanConfig(fa_keep=1, short_keep=0, skip_llm=True, skip_strategies=True)
    )

    assert result["watchlist"] == {"long": [], "short": []}
    assert result["fdr"] is None
    assert result["funnel"]["fdr_passed"] == 0
    assert result["funnel"]["watchlist"] == 0


def test_candidates_carry_cohort_for_both_slates(patched_pipeline) -> None:
    # short slate exercised end-to-end in test_scanner_tiers.py;
    # here we just prove every candidate dict has a "cohort" key and it is "long"
    # (BROKEN is the short FA pick but bars loading raises, so it never reaches
    # the candidates list — the fixture universe is entirely absorbed by the
    # long gate at fa_keep=3).
    result = patched_pipeline.run_scan(ScanConfig(fa_keep=3, skip_llm=True))
    assert result["candidates"], "fixture should produce at least one candidate"
    assert all("cohort" in c for c in result["candidates"])
    assert {c["cohort"] for c in result["candidates"]} <= {"long", "short"}


def test_short_cohort_candidate_dispatched_and_tagged(
    patched_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After EDGAR enrichment (default), BROKEN is the short FA pick at short_keep=1
    # (pass-1 prelim doubles the slates so BROKEN enters the short side; after
    # pass-2 re-ranking it remains the worst-fundamental name). The fixture
    # patched_pipeline raises for BROKEN on bars load, so:
    #   1. Serve bars for BROKEN via an overriding load_daily patch.
    #   2. Patch get_earnings_dates to return a UTC-localized empty frame for BROKEN
    #      (the fixture's fake_earnings returns tz-naive empty, which causes a timezone
    #      mismatch in the earnings comparison inside the bars try-block).
    #   3. Monkeypatch detect_all to return a real SetupSignal for BROKEN/short.
    from tradinglib.scanner import pipeline, setups
    from tradinglib.scanner.setups import SetupSignal

    detect_calls: list[tuple[str, str]] = []
    real_detect = setups.detect_all
    broken_bars = _quiet_bars("BROKEN")

    original_load = patched_pipeline.load_daily

    def load_with_broken(symbol, start=None, end=None, *, refresh=False):
        if symbol == "BROKEN":
            return broken_bars
        return original_load(symbol, start=start, end=end, refresh=refresh)

    monkeypatch.setattr(pipeline, "load_daily", load_with_broken)

    original_earnings = patched_pipeline.get_earnings_dates

    def earnings_with_broken(tickers, start=None, end=None, *, refresh=False):
        rows = original_earnings(tickers, start=start, end=end, refresh=refresh)
        # ensure tz-aware column so the upcoming-earnings comparison doesn't raise
        if "earnings_datetime" in rows.columns and len(rows) == 0:
            rows = rows.astype({"earnings_datetime": "datetime64[ns, UTC]"})
        return rows

    monkeypatch.setattr(pipeline, "get_earnings_dates", earnings_with_broken)

    def recording_detect(bars, *, stance="long", **kwargs):
        ticker = str(bars["symbol"].iloc[0]) if "symbol" in bars.columns else "?"
        detect_calls.append((ticker, stance))
        if ticker == "BROKEN" and stance == "short":
            return [
                SetupSignal(
                    ticker="BROKEN",
                    setup_type="base_breakdown",
                    score=0.75,
                    asof="2026-06-12",
                    trigger_level=98.0,
                    stop_level=102.0,
                    evidence={"atr": 1.0},
                )
            ]
        return real_detect(bars, stance=stance, **kwargs)

    monkeypatch.setattr(pipeline, "detect_all", recording_detect)

    result = patched_pipeline.run_scan(
        ScanConfig(fa_keep=1, short_keep=1, skip_llm=True, skip_strategies=True)
    )

    short_cands = [c for c in result["candidates"] if c.get("cohort") == "short"]
    assert short_cands, "BROKEN should appear as a short candidate when bars are served"
    assert short_cands[0]["ticker"] == "BROKEN"
    assert short_cands[0]["cohort"] == "short"
    # detect_all must have been called with stance="short" for BROKEN
    assert ("BROKEN", "short") in detect_calls


def test_fdr_wiring_unmocked(patched_pipeline) -> None:
    # No apply_fdr patch: proves the real call-site signature and alpha pass-through.
    #
    # Family m=1 (fa_keep=1, short_keep=0 → only DRIFT enters as a long candidate;
    # all 3 fixture tickers share identical fundamentals so rank(method="first")
    # assigns DRIFT rank=1 first):
    #   p = 1 - 0.95 = 0.05
    #   BH rank 1: 0.05 <= (1/1)*0.10 = 0.10  → PASS
    #   threshold = 0.05  (the p-value at the last passing rank)
    config = ScanConfig(fa_keep=1, short_keep=0, skip_llm=True)
    result = patched_pipeline.run_scan(config)

    assert result["fdr"]["alpha"] == 0.10
    assert result["fdr"]["family"] == 1
    assert result["fdr"]["threshold"] == pytest.approx(0.05)
    assert isinstance(result["fdr"]["threshold"], float)
    assert result["funnel"]["fdr_passed"] == 1


@pytest.mark.parametrize("status", ["waiting", "open"])
def test_recent_ledger_suppresses_active_campaign(fdr_pipeline, status: str) -> None:
    """Statuses 'waiting' and 'open' must suppress re-issuance."""
    recent = {
        "tickets": [
            {
                "ticker": "DRIFT",
                "stance": "long",
                "strategy": "sma_cross",
                "tier": "ticket",
                "status": status,
            }
        ]
    }
    result = fdr_pipeline.run_scan(
        ScanConfig(fa_keep=2, short_keep=0, skip_llm=True), recent_ledger=recent
    )
    issued = {(t["ticker"], t["stance"]) for t in result["tickets"]["long"]}
    assert ("DRIFT", "long") not in issued
    suppressed = result["suppressed"]
    assert any(
        s["ticker"] == "DRIFT" and s["tier"] == "ticket" and s["reason"] == "open campaign"
        for s in suppressed
    )
    assert result["funnel"]["suppressed"] == len(suppressed) >= 1


@pytest.mark.parametrize("status", ["stopped", "target", "expired", "error"])
def test_recent_ledger_closed_campaign_reissues(fdr_pipeline, status: str) -> None:
    """Statuses 'stopped', 'target', 'expired', 'error' must NOT block re-issuance."""
    recent = {
        "tickets": [
            {
                "ticker": "DRIFT",
                "stance": "long",
                "strategy": "sma_cross",
                "tier": "ticket",
                "status": status,
            }
        ]
    }
    result = fdr_pipeline.run_scan(
        ScanConfig(fa_keep=2, short_keep=0, skip_llm=True), recent_ledger=recent
    )
    issued = {(t["ticker"], t["stance"]) for t in result["tickets"]["long"]}
    assert ("DRIFT", "long") in issued
    assert result["suppressed"] == []


def test_recent_ledger_watch_tier_suppresses(fdr_pipeline) -> None:
    """A watch-tier row (status='waiting') must suppress the matching watchlist entry."""
    # fdr_pipeline demotes QUIET to the watchlist (fails FDR). Inject a ledger
    # row that matches QUIET's watchlist identity so it gets suppressed.
    # build_watchlist assigns strategy from the tournament winner; in our fixture
    # that is "sma_cross". The tier for a watchlist row is "watch".
    recent = {
        "tickets": [
            {
                "ticker": "QUIET",
                "stance": "long",
                "strategy": "sma_cross",
                "tier": "watch",
                "status": "waiting",
            }
        ]
    }
    result = fdr_pipeline.run_scan(
        ScanConfig(fa_keep=2, short_keep=0, skip_llm=True), recent_ledger=recent
    )
    wl_tickers = {r["ticker"] for r in result["watchlist"].get("long", [])}
    assert "QUIET" not in wl_tickers, "watch-tier waiting campaign must suppress watchlist row"
    suppressed = result["suppressed"]
    assert any(s["ticker"] == "QUIET" and s["tier"] == "watch" for s in suppressed), (
        f"expected QUIET watch suppression, got: {suppressed}"
    )


def test_recent_ledger_schema_round_trip(fdr_pipeline, tmp_path, monkeypatch) -> None:
    """A ledger built from real build_ledger output (status='waiting') must suppress
    the matching campaign — pins the vocabulary coupling between ledger.py and pipeline.py."""
    import json

    import pandas as pd

    from tradinglib.scanner.ledger import build_ledger

    # Build a minimal scan report directory that build_ledger can read.
    # The ticket must match what fdr_pipeline produces for DRIFT long.
    ticket = {
        "ticker": "DRIFT",
        "stance": "long",
        "strategy": "sma_cross",
        "tier": "ticket",
        "levels": {"entry": 100.0, "entry_type": "stop", "stop": 95.0, "target": 110.0},
    }
    report = {
        "asof": "2026-06-11",
        "funnel": {},
        "candidates": [],
        "errors": [],
        "tickets": {"long": [ticket], "short": []},
    }
    scan_dir = tmp_path / "scans" / "2026-06-11"
    scan_dir.mkdir(parents=True)
    (scan_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    # Provide bars where entry never triggers so status stays "waiting".
    idx = pd.date_range("2026-06-12", periods=3, freq="B", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [90.0, 91.0, 92.0],
            "high": [91.0, 92.0, 93.0],
            "low": [89.0, 90.0, 91.0],
            "close": [90.0, 91.0, 92.0],
        },
        index=idx,
    )
    ledger = build_ledger(tmp_path / "scans", asof="2026-06-12", loader=lambda t, **kw: bars)

    # Confirm the ledger produced a "waiting" row for DRIFT.
    waiting = [r for r in ledger["tickets"] if r["ticker"] == "DRIFT" and r["status"] == "waiting"]
    assert waiting, f"expected a waiting DRIFT row, got: {ledger['tickets']}"

    # Now run_scan with this real ledger as recent_ledger must suppress DRIFT.
    result = fdr_pipeline.run_scan(
        ScanConfig(fa_keep=2, short_keep=0, skip_llm=True), recent_ledger=ledger
    )
    issued = {(t["ticker"], t["stance"]) for t in result["tickets"]["long"]}
    assert ("DRIFT", "long") not in issued, "build_ledger 'waiting' row must suppress re-issuance"
    assert any(s["ticker"] == "DRIFT" for s in result["suppressed"])


def test_active_campaigns_tolerates_malformed_rows(fdr_pipeline) -> None:
    """A ledger row missing identity keys must not cause a KeyError; valid rows still suppress."""
    recent = {
        "tickets": [
            # malformed: status open but no 'strategy' key
            {"ticker": "DRIFT", "stance": "long", "status": "open"},
            # valid: this must still suppress
            {
                "ticker": "DRIFT",
                "stance": "long",
                "strategy": "sma_cross",
                "tier": "ticket",
                "status": "open",
            },
        ]
    }
    result = fdr_pipeline.run_scan(
        ScanConfig(fa_keep=2, short_keep=0, skip_llm=True), recent_ledger=recent
    )
    # Scan must complete (no KeyError).
    issued = {(t["ticker"], t["stance"]) for t in result["tickets"]["long"]}
    assert ("DRIFT", "long") not in issued, (
        "valid row must still suppress despite malformed sibling"
    )


def test_regime_gate_suppresses_meanrev_ticket_in_downtrend(
    fdr_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """regime_gate=True + down-trend must suppress the would-be ticket."""
    import tradinglib.scanner.regime as regime_module

    # The fixture's winner strategy is sma_cross (trend), not meanrev.
    # Reclassify it so gate_reason fires.
    monkeypatch.setitem(regime_module.STRATEGY_STYLES, "sma_cross", "meanrev")
    # Force a down-trend regardless of the fake SPY bars.
    monkeypatch.setattr(fdr_pipeline, "regime_state", lambda close: {"trend": "down"})

    result = fdr_pipeline.run_scan(
        ScanConfig(fa_keep=2, short_keep=0, skip_llm=True, regime_gate=True)
    )

    # DRIFT ticket must be suppressed
    assert ("DRIFT", "long") not in {(t["ticker"], t["stance"]) for t in result["tickets"]["long"]}
    suppressed = result["suppressed"]
    assert any(
        s["ticker"] == "DRIFT" and s["tier"] == "ticket" and s["reason"].startswith("meanrev long")
        for s in suppressed
    ), f"expected DRIFT ticket suppression, got: {suppressed}"
    assert result["regime"]["trend"] == "down"


def test_regime_gate_off_by_default(fdr_pipeline) -> None:
    """regime_gate defaults to False: regime is always reported, issuance unchanged."""
    result = fdr_pipeline.run_scan(ScanConfig(fa_keep=2, short_keep=0, skip_llm=True))

    assert "regime" in result
    # Gate is off — DRIFT must still get a ticket
    issued = {(t["ticker"], t["stance"]) for t in result["tickets"]["long"]}
    assert ("DRIFT", "long") in issued
