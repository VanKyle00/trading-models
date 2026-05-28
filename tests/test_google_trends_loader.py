"""Tests for the Google Trends loader.

Network access is stubbed out — the tests exercise canonicalization,
caching, and the partial-row filter against a synthetic pytrends response.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def fake_trends_frame() -> pd.DataFrame:
    """Mimic a pytrends weekly response, including an isPartial row."""
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2021-01-03", "2021-01-10", "2021-01-17", "2021-01-24", "2021-01-31"]),
        name="date",
    )
    return pd.DataFrame(
        {
            "bitcoin": [89, 87, 55, 48, 72],
            "isPartial": [False, False, False, False, True],
        },
        index=idx,
    )


def _patched_trendreq(frame: pd.DataFrame) -> MagicMock:
    """Build a MagicMock that mimics TrendReq's two-step usage."""
    inst = MagicMock()
    inst.build_payload = MagicMock()
    inst.interest_over_time = MagicMock(return_value=frame)
    return inst


def test_load_drops_partial_rows(
    fake_trends_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.sentiment import google_trends as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    with patch.object(loader, "TrendReq", return_value=_patched_trendreq(fake_trends_frame)):
        series = loader.load_interest("bitcoin", timeframe="2021-01-01 2021-02-01")

    assert len(series) == 4  # the isPartial=True row is filtered
    assert series.name == "bitcoin"
    assert str(series.index.tz) == "UTC"


def test_load_writes_cache(
    fake_trends_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.sentiment import google_trends as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    with patch.object(loader, "TrendReq", return_value=_patched_trendreq(fake_trends_frame)):
        loader.load_interest("bitcoin", timeframe="2021-01-01 2021-02-01")

    expected = (
        tmp_path / "google_trends" / "bitcoin__2021-01-01_2021-02-01__global" / "interest.parquet"
    )
    assert expected.exists()


def test_load_uses_cache_on_second_call(
    fake_trends_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradinglib.loaders.sentiment import google_trends as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    trendreq_mock = _patched_trendreq(fake_trends_frame)
    with patch.object(loader, "TrendReq", return_value=trendreq_mock) as patched_cls:
        loader.load_interest("bitcoin", timeframe="2021-01-01 2021-02-01")
        loader.load_interest("bitcoin", timeframe="2021-01-01 2021-02-01")
        # First call constructs TrendReq; second call must read the cache
        assert patched_cls.call_count == 1


def test_load_raises_on_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tradinglib.loaders.sentiment import google_trends as loader

    monkeypatch.setattr(loader, "processed_dir", lambda source: tmp_path / source)
    empty = pd.DataFrame()
    with (
        patch.object(loader, "TrendReq", return_value=_patched_trendreq(empty)),
        pytest.raises(ValueError, match="no data"),
    ):
        loader.load_interest("zzzznonsense", timeframe="2021-01-01 2021-02-01")
