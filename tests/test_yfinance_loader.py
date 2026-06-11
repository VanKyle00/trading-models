"""Tests for the yfinance loader.

The download path hits the network, so these tests stub out ``yf.download``
and exercise the canonicalization + caching logic against a synthetic
response.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def fake_yf_frame() -> pd.DataFrame:
    """Mimic yfinance's MultiIndex-column output for a single symbol."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            ("Open", "SPY"): [100.0, 101.0, 102.0, 103.0, 104.0],
            ("High", "SPY"): [101.0, 102.0, 103.0, 104.0, 105.0],
            ("Low", "SPY"): [99.0, 100.0, 101.0, 102.0, 103.0],
            ("Close", "SPY"): [100.5, 101.5, 102.5, 103.5, 104.5],
            ("Volume", "SPY"): [1_000_000, 1_100_000, 900_000, 1_200_000, 1_050_000],
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def test_canonicalize_flattens_and_renames(fake_yf_frame: pd.DataFrame) -> None:
    from tradinglib.loaders.equities.yfinance import _canonicalize

    out = _canonicalize(fake_yf_frame, "SPY")
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "symbol"]
    assert str(out.index.tz) == "UTC"
    assert out["symbol"].iloc[0] == "SPY"
    assert out["volume"].dtype == "int64"


def test_load_daily_writes_cache(
    fake_yf_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    # Redirect processed_dir to a tmp location
    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "download", return_value=fake_yf_frame):
        df = loader.load_daily("SPY")

    cached = tmp_path / "yfinance" / "SPY" / "daily.parquet"
    assert cached.exists()
    assert len(df) == 5
    assert df["symbol"].iloc[0] == "SPY"


def test_load_daily_uses_cache_on_second_call(
    fake_yf_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "download", return_value=fake_yf_frame) as mock_dl:
        loader.load_daily("SPY")
        loader.load_daily("SPY")
        # Second call must hit the cache, not the network
        assert mock_dl.call_count == 1


def test_load_daily_filters_by_date_range(
    fake_yf_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "download", return_value=fake_yf_frame):
        df = loader.load_daily("SPY", start="2024-01-02", end="2024-01-04")

    assert len(df) == 3
    assert df.index.min() == pd.Timestamp("2024-01-02", tz="UTC")
    assert df.index.max() == pd.Timestamp("2024-01-04", tz="UTC")


def test_load_daily_raises_on_empty_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with (
        patch.object(loader.yf, "download", return_value=pd.DataFrame()),
        pytest.raises(ValueError, match="no data"),
    ):
        loader.load_daily("ZZZZ")


@pytest.fixture
def fake_yf_frame_with_nan_tail(fake_yf_frame: pd.DataFrame) -> pd.DataFrame:
    """yfinance glitch: a trailing row with volume populated but NaN prices."""
    df = fake_yf_frame.copy()
    nan_day = pd.Timestamp("2024-01-06")
    for col in ("Open", "High", "Low", "Close"):
        df.loc[nan_day, (col, "SPY")] = float("nan")
    df.loc[nan_day, ("Volume", "SPY")] = 500_000
    return df


def test_load_daily_drops_nan_price_rows(
    fake_yf_frame_with_nan_tail: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    with patch.object(loader.yf, "download", return_value=fake_yf_frame_with_nan_tail):
        df = loader.load_daily("SPY")

    assert len(df) == 5  # the NaN-price tail row is gone
    assert df.index.max() == pd.Timestamp("2024-01-05", tz="UTC")
    assert not df[["open", "high", "low", "close"]].isna().any().any()


def test_load_daily_filters_nan_rows_from_poisoned_cache(
    fake_yf_frame_with_nan_tail: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    # A cache written before the guard existed still contains the bad row.
    poisoned = loader._canonicalize(fake_yf_frame_with_nan_tail, "SPY")
    out = tmp_path / "yfinance" / "SPY" / "daily.parquet"
    out.parent.mkdir(parents=True)
    poisoned.to_parquet(out)

    df = loader.load_daily("SPY")  # cache hit — no download stub needed
    assert len(df) == 5
    assert not df[["open", "high", "low", "close"]].isna().any().any()
