"""Tests for the swing-setup detectors (Stage 2 of the scanner)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.scanner.setups import (
    SetupSignal,
    detect_all,
    detect_base_breakdown,
    detect_base_breakout,
    detect_ma_pullback,
    detect_ma_rally_fade,
    detect_pead,
)


def _bars(
    close: np.ndarray,
    volume: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    symbol: str = "TEST",
) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(close), freq="B", tz="UTC")
    close_s = pd.Series(close, index=idx, dtype=float)
    high_s = close_s * 1.01 if high is None else pd.Series(high, index=idx, dtype=float)
    low_s = close_s * 0.99 if low is None else pd.Series(low, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "volume": pd.Series(volume, index=idx, dtype=float),
            "symbol": symbol,
        },
        index=idx,
    )


def _breakout_bars() -> pd.DataFrame:
    # 200-bar uptrend 50 -> 100, then a 100-bar base around 97 whose range
    # narrows from +/-3 to +/-0.5 while volume dries up.
    trend = np.linspace(50.0, 100.0, 200)
    amp = np.linspace(3.0, 0.5, 100)
    base = 97.0 + amp * np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    close = np.concatenate([trend, base])
    volume = np.concatenate([np.full(200, 2_000_000.0), np.linspace(1_500_000, 800_000, 100)])
    return _bars(close, volume)


def _breakdown_bars() -> pd.DataFrame:
    # Mirror of _breakout_bars: 200-bar downtrend 100 -> 50, then a 100-bar
    # base around 52 whose range narrows while volume dries up.
    trend = np.linspace(100.0, 50.0, 200)
    amp = np.linspace(3.0, 0.5, 100)
    base = 52.0 + amp * np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    close = np.concatenate([trend, base])
    volume = np.concatenate([np.full(200, 2_000_000.0), np.linspace(1_500_000, 800_000, 100)])
    return _bars(close, volume)


def _pullback_bars() -> pd.DataFrame:
    # Long gentle uptrend, a strong 56-bar leg, then a 24-bar orderly
    # pullback (alternating -1.4% / +0.8%) into the rising 50-day MA.
    gentle = 100.0 * 1.002 ** np.arange(220)
    leg = gentle[-1] * 1.006 ** np.arange(1, 57)
    pull = [leg[-1]]
    for i in range(24):
        step = -0.014 if i % 2 == 0 else 0.008
        pull.append(pull[-1] * (1 + step))
    close = np.concatenate([gentle, leg, np.array(pull[1:])])
    volume = np.concatenate([np.full(276, 1_000_000.0), np.full(24, 800_000.0)])
    return _bars(close, volume)


def _rally_fade_bars() -> pd.DataFrame:
    # Mirror of _pullback_bars: gentle downtrend, a steep 56-bar down-leg,
    # then a 24-bar orderly rally (alternating +1.4% / -0.8%) into the
    # declining 50-day MA.
    gentle = 100.0 * 0.998 ** np.arange(220)
    leg = gentle[-1] * 0.994 ** np.arange(1, 57)
    rally = [leg[-1]]
    for i in range(24):
        step = 0.014 if i % 2 == 0 else -0.008
        rally.append(rally[-1] * (1 + step))
    close = np.concatenate([gentle, leg, np.array(rally[1:])])
    volume = np.concatenate([np.full(276, 1_000_000.0), np.full(24, 800_000.0)])
    return _bars(close, volume)


def _pead_bars() -> tuple[pd.DataFrame, pd.Timestamp]:
    # Flat at 100, an 8% earnings-day pop on 3x volume at bar 110, then a
    # gentle upward drift that holds above the earnings-day low.
    close = np.concatenate([np.full(110, 100.0), [108.0], np.linspace(108.2, 110.0, 9)])
    volume = np.concatenate([np.full(110, 1_000_000.0), [3_000_000.0], np.full(9, 1_200_000.0)])
    low = np.concatenate([np.full(110, 99.0), [105.0], np.linspace(107.0, 108.8, 9)])
    high = close * 1.01
    bars = _bars(close, volume, high=high, low=low)
    return bars, bars.index[110]


def test_base_breakout_detected() -> None:
    signal = detect_base_breakout(_breakout_bars())

    assert isinstance(signal, SetupSignal)
    assert signal.setup_type == "base_breakout"
    assert signal.ticker == "TEST"
    assert 0.0 <= signal.score <= 1.0
    assert signal.trigger_level > signal.stop_level


def test_base_breakout_rejects_beaten_down_stock() -> None:
    # Same base shape but 40% below the 52-week high: not a breakout candidate.
    trend = np.linspace(150.0, 90.0, 200)
    amp = np.linspace(3.0, 0.5, 100)
    base = 88.0 + amp * np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    close = np.concatenate([trend, base])
    volume = np.concatenate([np.full(200, 2_000_000.0), np.linspace(1_500_000, 800_000, 100)])

    assert detect_base_breakout(_bars(close, volume)) is None


def test_base_breakdown_detected() -> None:
    signal = detect_base_breakdown(_breakdown_bars())

    assert isinstance(signal, SetupSignal)
    assert signal.setup_type == "base_breakdown"
    assert 0.0 <= signal.score <= 1.0
    assert signal.trigger_level < signal.stop_level  # short: stop ABOVE trigger


def test_base_breakdown_rejects_stock_near_its_high() -> None:
    # The long-side breakout fixture sits near its 52-week HIGH: not breakdown material.
    assert detect_base_breakdown(_breakout_bars()) is None


def test_base_breakout_requires_history() -> None:
    short = _breakout_bars().iloc[-100:]
    assert detect_base_breakout(short) is None


def test_ma_pullback_detected() -> None:
    signal = detect_ma_pullback(_pullback_bars())

    assert isinstance(signal, SetupSignal)
    assert signal.setup_type == "ma_pullback"
    assert 0.0 <= signal.score <= 1.0
    assert signal.trigger_level > signal.stop_level


def test_ma_pullback_rejects_downtrend() -> None:
    close = 200.0 * 0.999 ** np.arange(300)
    volume = np.full(300, 1_000_000.0)

    assert detect_ma_pullback(_bars(close, volume)) is None


def test_pead_detected() -> None:
    bars, earnings_dt = _pead_bars()

    signal = detect_pead(bars, [earnings_dt])

    assert isinstance(signal, SetupSignal)
    assert signal.setup_type == "pead"
    assert 0.0 <= signal.score <= 1.0
    # stop sits at the earnings-day low
    assert signal.stop_level == 105.0


def test_pead_rejects_stale_earnings() -> None:
    bars, _ = _pead_bars()
    stale = bars.index[80]  # 39 bars ago — outside the 3-15 day window

    assert detect_pead(bars, [stale]) is None


def test_pead_rejects_small_move() -> None:
    close = np.concatenate([np.full(110, 100.0), [101.5], np.full(9, 101.8)])
    volume = np.concatenate([np.full(110, 1_000_000.0), [3_000_000.0], np.full(9, 1_200_000.0)])
    bars = _bars(close, volume)

    assert detect_pead(bars, [bars.index[110]]) is None


def test_detect_all_collects_signals() -> None:
    bars, earnings_dt = _pead_bars()

    signals = detect_all(bars, earnings_datetimes=[earnings_dt])

    assert [s.setup_type for s in signals] == ["pead"]


def test_detect_all_empty_for_quiet_stock() -> None:
    close = 100.0 + np.sin(np.arange(300) / 5.0)
    volume = np.full(300, 1_000_000.0)

    assert detect_all(_bars(close, volume)) == []


def test_ma_rally_fade_detected() -> None:
    signal = detect_ma_rally_fade(_rally_fade_bars())

    assert isinstance(signal, SetupSignal)
    assert signal.setup_type == "ma_rally_fade"
    assert signal.trigger_level < signal.stop_level


def test_ma_rally_fade_rejects_uptrend() -> None:
    assert detect_ma_rally_fade(_pullback_bars()) is None
