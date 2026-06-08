"""Tests for the yfinance earnings-calendar loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def fake_earnings_frame() -> pd.DataFrame:
    """Mimic yfinance Ticker.get_earnings_dates(): tz-aware DatetimeIndex named
    'Earnings Date', plus estimate/actual columns we ignore."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-10-31 16:05:00", tz="America/New_York"),  # AMC
            pd.Timestamp("2024-07-25 08:30:00", tz="America/New_York"),  # BMO
        ],
        name="Earnings Date",
    )
    return pd.DataFrame({"EPS Estimate": [1.0, 0.9], "Reported EPS": [1.1, 0.95]}, index=idx)


def test_canonicalize_columns_and_sessions(fake_earnings_frame: pd.DataFrame) -> None:
    from tradinglib.loaders.events import earnings as loader

    out = loader._canonicalize(fake_earnings_frame, "AAPL")

    assert list(out.columns) == ["ticker", "earnings_datetime", "session"]
    assert (out["ticker"] == "AAPL").all()
    assert str(out["earnings_datetime"].dt.tz) == "UTC"
    # 16:05 ET -> after close -> amc; 08:30 ET -> before open -> bmo
    assert set(out["session"]) == {"amc", "bmo"}


def test_get_earnings_dates_caches_and_filters(
    fake_earnings_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.events import earnings as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame:
            return fake_earnings_frame

    with patch.object(loader.yf, "Ticker", _FakeTicker):
        df = loader.get_earnings_dates(["AAPL"], start="2024-01-01", end="2024-12-31")

    cached = list((tmp_path / "events" / "earnings" / "AAPL").glob("*.parquet"))
    assert len(cached) == 1
    assert set(df["ticker"]) == {"AAPL"}
    assert len(df) == 2


def test_get_earnings_dates_handles_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.events import earnings as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    class _EmptyTicker:
        def __init__(self, symbol: str) -> None:
            pass

        def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame | None:
            return None

    with patch.object(loader.yf, "Ticker", _EmptyTicker):
        df = loader.get_earnings_dates(["ZZZZ"], start="2024-01-01", end="2024-12-31")

    assert list(df.columns) == ["ticker", "earnings_datetime", "session"]
    assert df.empty


def test_get_earnings_dates_mixed_empty_and_nonempty_keeps_dtype(
    fake_earnings_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.events import earnings as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    class _MixedTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def get_earnings_dates(self, limit: int = 24) -> pd.DataFrame | None:
            return None if self.symbol == "ZZZZ" else fake_earnings_frame

    with patch.object(loader.yf, "Ticker", _MixedTicker):
        df = loader.get_earnings_dates(["AAPL", "ZZZZ"], start="2024-01-01", end="2024-12-31")

    # tz-aware dtype must survive concat so the date filter does not raise
    assert str(df["earnings_datetime"].dt.tz) == "UTC"
    assert set(df["ticker"]) == {"AAPL"}
    assert len(df) == 2
