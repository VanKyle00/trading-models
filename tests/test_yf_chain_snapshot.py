"""Tests for the yfinance forward chain-snapshot collector (yfinance mocked, never live)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_COLUMNS = ["date", "ticker", "expiration", "strike", "right", "bid", "ask", "iv", "spot"]


def _fake_leg(strikes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": strikes,
            "bid": [1.0] * len(strikes),
            "ask": [1.2] * len(strikes),
            "impliedVolatility": [0.30] * len(strikes),
            "openInterest": [500] * len(strikes),
            "volume": [25] * len(strikes),
        }
    )


class _FakeTicker:
    """Two expirations: one inside the 45-day window, one far beyond it."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        self.options = (
            (today + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            (today + pd.Timedelta(days=200)).strftime("%Y-%m-%d"),
        )
        self.fast_info = {"lastPrice": 100.0}

    def option_chain(self, expiration: str) -> SimpleNamespace:
        return SimpleNamespace(calls=_fake_leg([95.0, 100.0]), puts=_fake_leg([95.0, 100.0]))


def test_snapshot_writes_canonical_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)

    counts = yf_chain.snapshot_chains(["AAPL"])

    assert counts["AAPL"] == 4  # 1 in-window expiration x 2 strikes x 2 rights
    files = list((tmp_path / "options" / "yf_snapshots" / "AAPL").glob("*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert list(df.columns) == _COLUMNS
    assert set(df["right"]) == {"call", "put"}
    assert (df["spot"] == 100.0).all()
    # the 200-day expiration is beyond the 45-day cutoff and must be absent
    assert df["expiration"].nunique() == 1


def test_snapshot_is_idempotent_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)

    first = yf_chain.snapshot_chains(["AAPL"])
    second = yf_chain.snapshot_chains(["AAPL"])

    assert first["AAPL"] == 4
    assert second["AAPL"] == -1  # sentinel: already snapshotted today, no refetch


def test_one_failing_ticker_does_not_abort_the_watchlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)

    class _FlakyTicker(_FakeTicker):
        def __init__(self, symbol: str) -> None:
            if symbol == "BAD":
                raise RuntimeError("yfinance flake")
            super().__init__(symbol)

    monkeypatch.setattr(yf_chain.yf, "Ticker", _FlakyTicker)

    counts = yf_chain.snapshot_chains(["AAPL", "BAD", "MSFT"])

    assert counts["AAPL"] == 4 and counts["MSFT"] == 4
    assert counts["BAD"] == -2
    assert not (tmp_path / "options" / "yf_snapshots" / "BAD").exists()


def test_empty_options_not_persisted_so_rerun_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)

    class _NoOptionsTicker(_FakeTicker):
        def __init__(self, symbol: str) -> None:
            super().__init__(symbol)
            self.options = ()

    monkeypatch.setattr(yf_chain.yf, "Ticker", _NoOptionsTicker)
    first = yf_chain.snapshot_chains(["AAPL"])
    assert first["AAPL"] == 0
    assert not list((tmp_path / "options").rglob("*.parquet"))  # nothing locked in

    # a later re-run with working data must succeed (not -1)
    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)
    second = yf_chain.snapshot_chains(["AAPL"])
    assert second["AAPL"] == 4


def test_snapshot_roundtrip_preserves_dtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain, "processed_dir", lambda source: tmp_path / source)
    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)

    yf_chain.snapshot_chains(["AAPL"])
    files = list((tmp_path / "options" / "yf_snapshots" / "AAPL").glob("*.parquet"))
    df = pd.read_parquet(files[0])

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_datetime64_any_dtype(df["expiration"])
    for col in ("strike", "bid", "ask", "iv", "spot"):
        assert pd.api.types.is_float_dtype(df[col]), col


def test_fetch_chain_returns_extended_schema_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    monkeypatch.setattr(yf_chain.yf, "Ticker", _FakeTicker)

    df = yf_chain.fetch_chain("AAPL")

    assert list(df.columns) == yf_chain.CHAIN_COLUMNS
    assert (df["open_interest"] == 500).all()
    assert (df["volume"] == 25).all()
    # of the two fake expirations (10d and 200d), only 10d sits inside the 120-day window
    assert df["expiration"].nunique() == 1
    assert len(df) == 4  # 1 expiration x 2 strikes x 2 rights


def test_fetch_chain_window_is_wider_than_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    class _MidTicker(_FakeTicker):
        def __init__(self, symbol: str) -> None:
            super().__init__(symbol)
            today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
            self.options = ((today + pd.Timedelta(days=80)).strftime("%Y-%m-%d"),)

    monkeypatch.setattr(yf_chain.yf, "Ticker", _MidTicker)

    df = yf_chain.fetch_chain("AAPL")  # default 120-day window
    assert df["expiration"].nunique() == 1  # 80 DTE is in for tickets, out for snapshots


def test_fetch_chain_empty_options_returns_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    class _NoOptionsTicker(_FakeTicker):
        def __init__(self, symbol: str) -> None:
            super().__init__(symbol)
            self.options = ()

    monkeypatch.setattr(yf_chain.yf, "Ticker", _NoOptionsTicker)

    df = yf_chain.fetch_chain("AAPL")
    assert df.empty
    assert list(df.columns) == yf_chain.CHAIN_COLUMNS


def test_fetch_chain_missing_oi_columns_yield_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.options import yf_chain

    class _BareTicker(_FakeTicker):
        def option_chain(self, expiration: str):  # legs without openInterest/volume
            from types import SimpleNamespace

            leg = _fake_leg([95.0, 100.0]).drop(columns=["openInterest", "volume"])
            return SimpleNamespace(calls=leg, puts=leg)

    monkeypatch.setattr(yf_chain.yf, "Ticker", _BareTicker)

    df = yf_chain.fetch_chain("AAPL")
    assert df["open_interest"].isna().all() and df["volume"].isna().all()
    # the NaN fill must stay numeric — an object column would break the liquidity gate
    assert pd.api.types.is_float_dtype(df["open_interest"])
    assert pd.api.types.is_float_dtype(df["volume"])
