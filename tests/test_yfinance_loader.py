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


@pytest.fixture
def fake_yf_unadjusted() -> pd.DataFrame:
    """auto_adjust=False output: RAW (here 2x, as if pre a 2:1 split) OHLC plus a
    separate 'Adj Close' column — what the second fetch returns."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            ("Open", "SPY"): [200.0, 202.0, 204.0, 206.0, 208.0],
            ("High", "SPY"): [202.0, 204.0, 206.0, 208.0, 210.0],
            ("Low", "SPY"): [198.0, 200.0, 202.0, 204.0, 206.0],
            ("Close", "SPY"): [201.0, 203.0, 205.0, 207.0, 209.0],
            ("Adj Close", "SPY"): [100.5, 101.5, 102.5, 103.5, 104.5],
            ("Volume", "SPY"): [1_000_000, 1_100_000, 900_000, 1_200_000, 1_050_000],
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def test_load_daily_persists_unadjusted_alongside_adjusted(
    fake_yf_frame: pd.DataFrame,
    fake_yf_unadjusted: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradinglib.loaders.equities import yfinance as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)

    # first fetch = adjusted (auto_adjust=True), second = unadjusted (auto_adjust=False)
    with patch.object(loader.yf, "download", side_effect=[fake_yf_frame, fake_yf_unadjusted]):
        df = loader.load_daily("SPY")

    assert {"unadj_open", "unadj_high", "unadj_low", "unadj_close"}.issubset(df.columns)
    # adjusted columns are byte-identical to the auto_adjust=True canonicalization
    assert df["close"].tolist() == [100.5, 101.5, 102.5, 103.5, 104.5]
    # unadjusted columns carry the RAW (pre-split) prices the ticket actually traded
    assert df["unadj_close"].tolist() == [201.0, 203.0, 205.0, 207.0, 209.0]
    assert df["unadj_open"].iloc[0] == 200.0
    # survives the parquet round-trip on the cache-hit read
    cached = loader.load_daily("SPY")
    assert cached["unadj_close"].tolist() == [201.0, 203.0, 205.0, 207.0, 209.0]


def test_attach_unadjusted_is_all_or_nothing_on_index_gaps() -> None:
    # If the auto_adjust=False fetch misses a trading day the adjusted fetch has,
    # reindex would punch NaN into unadj_*; the ledger then swaps that NaN into
    # open/high/low/close and simulate_ticket's dropna silently drops the bar AND
    # the scored_unadjusted=True suppresses the A5b fallback -> wrong R. So a gappy
    # unadjusted fetch must attach NOTHING (fall back to the adjusted/A5b path).
    from tradinglib.loaders.equities.yfinance import _attach_unadjusted, _canonicalize

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    adj = pd.DataFrame(
        {
            ("Open", "SPY"): [50.0, 51.0, 52.0],
            ("High", "SPY"): [50.5, 51.5, 52.5],
            ("Low", "SPY"): [49.5, 50.5, 51.5],
            ("Close", "SPY"): [50.0, 51.0, 52.0],
            ("Volume", "SPY"): [1, 2, 3],
        },
        index=idx,
    )
    base = _canonicalize(adj, "SPY")
    gappy = pd.DataFrame(  # missing 2024-01-02
        {
            ("Open", "SPY"): [100.0, 104.0],
            ("High", "SPY"): [101.0, 105.0],
            ("Low", "SPY"): [99.0, 103.0],
            ("Close", "SPY"): [100.0, 104.0],
        },
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"]),
    )
    out = _attach_unadjusted(base.copy(), gappy)
    assert "unadj_close" not in out.columns  # all-or-nothing: gap -> no attach
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "symbol"]


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
        after_first = mock_dl.call_count  # 2: the adjusted + unadjusted fetches
        loader.load_daily("SPY")
        # Second call must hit the cache, not the network — no new download calls
        assert mock_dl.call_count == after_first


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
