"""Tests for the tournament strategy registry — signal and levels math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.tournament.strategies import STRATEGIES
from tradinglib.validation.search import expand_grid


def _bars(
    close: np.ndarray, high: np.ndarray | None = None, low: np.ndarray | None = None
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2022-01-03", periods=len(close), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01 if high is None else np.asarray(high, dtype=float),
            "low": close * 0.99 if low is None else np.asarray(low, dtype=float),
            "close": close,
        },
        index=idx,
    )


def _trend_bars(n: int = 420, rate: float = 1.005) -> pd.DataFrame:
    return _bars(100.0 * rate ** np.arange(n))


def _flat_bars(n: int = 60, close: float = 100.0, spread: float = 1.0) -> pd.DataFrame:
    c = np.full(n, close)
    return _bars(c, high=c + spread, low=c - spread)


def test_sma_cross_long_rides_an_uptrend() -> None:
    bars = _trend_bars()
    train, test = bars.iloc[:300], bars.iloc[300:]
    sig = STRATEGIES["sma_cross"].make_signal(train, test, {"fast": 10, "slow": 50}, "long")
    assert sig.index.equals(test.index)
    assert set(np.unique(sig)) <= {0.0, 1.0}
    assert sig.iloc[-1] == 1.0  # fast SMA above slow on a steady rise


def test_sma_cross_short_is_the_mirror_rule_not_a_sign_flip() -> None:
    bars = _trend_bars(rate=0.995)  # downtrend
    train, test = bars.iloc[:300], bars.iloc[300:]
    short = STRATEGIES["sma_cross"].make_signal(train, test, {"fast": 10, "slow": 50}, "short")
    assert set(np.unique(short)) <= {-1.0, 0.0}
    assert short.iloc[-1] == -1.0
    long = STRATEGIES["sma_cross"].make_signal(train, test, {"fast": 10, "slow": 50}, "long")
    assert (long == 0.0).all()  # the long rule never fires in a downtrend


def test_make_signal_handles_in_sample_train_equals_test() -> None:
    # walk_forward scores in-sample via make_signal(train, train, params)
    bars = _trend_bars()
    train = bars.iloc[:300]
    sig = STRATEGIES["sma_cross"].make_signal(train, train, {"fast": 10, "slow": 50}, "long")
    assert sig.index.equals(train.index)  # concat deduped, no doubled index


def test_sma_cross_levels_atr_stop_and_2r_target() -> None:
    bars = _flat_bars()  # ATR(14) == 2.0
    lv = STRATEGIES["sma_cross"].levels(bars, {"fast": 10, "slow": 50}, "long")
    assert lv.entry == 100.0 and lv.entry_type == "market"
    assert lv.stop == 96.0  # entry - 2*ATR
    assert lv.target == 108.0  # entry + 2R
    short = STRATEGIES["sma_cross"].levels(bars, {"fast": 10, "slow": 50}, "short")
    assert short.stop == 104.0 and short.target == 92.0


def test_donchian_long_enters_on_breakout_and_exits_mid_channel() -> None:
    flat = np.full(80, 100.0)
    spike = np.array([106.0, 107.0, 96.0])  # breakout, hold, collapse through mid
    bars = _bars(np.concatenate([flat, spike]))
    train, test = bars.iloc[:60], bars.iloc[60:]
    sig = STRATEGIES["donchian"].make_signal(train, test, {"n": 20}, "long")
    assert sig.loc[bars.index[80]] == 1.0  # close 106 > prior 20d high (101)
    assert sig.loc[bars.index[81]] == 1.0  # still above mid-channel
    assert sig.loc[bars.index[82]] == 0.0  # 96 below mid-channel -> exit


def test_donchian_short_sells_the_low_break() -> None:
    flat = np.full(80, 100.0)
    bars = _bars(np.concatenate([flat, [93.0, 92.0]]))
    train, test = bars.iloc[:60], bars.iloc[60:]
    sig = STRATEGIES["donchian"].make_signal(train, test, {"n": 20}, "short")
    assert sig.loc[bars.index[80]] == -1.0  # close 93 < prior 20d low (99)
    assert sig.loc[bars.index[81]] == -1.0


def test_donchian_levels_stop_at_mid_channel() -> None:
    bars = _flat_bars()
    lv = STRATEGIES["donchian"].levels(bars, {"n": 20}, "long")
    assert lv.entry == 101.0 and lv.entry_type == "stop"  # buy-stop at the 20d high
    assert lv.stop == 100.0  # mid-channel is the native exit
    assert lv.target == 103.0  # entry + 2R
    short = STRATEGIES["donchian"].levels(bars, {"n": 20}, "short")
    assert short.entry == 99.0 and short.stop == 100.0 and short.target == 97.0


def test_donchian_levels_reject_degenerate_channel() -> None:
    bars = _flat_bars(spread=0.0)  # zero-width channel
    with pytest.raises(ValueError, match="channel"):
        STRATEGIES["donchian"].levels(bars, {"n": 20}, "long")


def test_rsi2_long_buys_the_dip_in_an_uptrend_and_exits_above_sma5() -> None:
    rise = 100.0 * 1.004 ** np.arange(280)
    dipped = rise.copy()
    dipped[260] = rise[260] * 0.97  # two-bar slide pins RSI(2) near 0
    dipped[261] = dipped[260] * 0.97
    dipped[262] = dipped[261] * 1.06  # snap back above SMA(5)
    bars = _bars(dipped)
    train, test = bars.iloc[:250], bars.iloc[250:]
    sig = STRATEGIES["rsi2"].make_signal(train, test, {"entry_thr": 10}, "long")
    assert sig.loc[bars.index[261]] == 1.0  # oversold inside the uptrend
    assert sig.loc[bars.index[262]] == 0.0  # snapped back above SMA(5) -> exit


def test_rsi2_short_fades_the_pop_in_a_downtrend() -> None:
    fall = 100.0 * 0.996 ** np.arange(280)
    popped = fall.copy()
    popped[260] = fall[260] * 1.03  # two-bar pop pins RSI(2) near 100
    popped[261] = popped[260] * 1.03
    bars = _bars(popped)
    train, test = bars.iloc[:250], bars.iloc[250:]
    sig = STRATEGIES["rsi2"].make_signal(train, test, {"entry_thr": 10}, "short")
    assert sig.loc[bars.index[261]] == -1.0  # overbought pop inside the downtrend


def test_rsi2_levels_market_entry_with_atr_stop() -> None:
    bars = _flat_bars()
    lv = STRATEGIES["rsi2"].levels(bars, {"entry_thr": 10}, "long")
    assert lv.entry == 100.0 and lv.entry_type == "market"
    assert lv.stop == 96.0 and lv.target == 108.0
    assert "RSI(2)" in lv.condition and "SMA(5)" in lv.condition


def test_macd_long_in_uptrend_short_in_downtrend() -> None:
    params = {"fast": 12, "slow": 26, "signal": 9}
    up = _trend_bars()
    sig = STRATEGIES["macd"].make_signal(up.iloc[:300], up.iloc[300:], params, "long")
    assert set(np.unique(sig)) <= {0.0, 1.0}
    assert sig.iloc[-1] == 1.0
    down = _trend_bars(rate=0.995)
    short = STRATEGIES["macd"].make_signal(down.iloc[:300], down.iloc[300:], params, "short")
    assert short.iloc[-1] == -1.0


def test_macd_levels_market_entry_with_atr_stop() -> None:
    bars = _flat_bars()
    lv = STRATEGIES["macd"].levels(bars, {"fast": 12, "slow": 26, "signal": 9}, "long")
    assert lv.entry == 100.0 and lv.entry_type == "market"
    assert lv.stop == 96.0 and lv.target == 108.0


def test_bollinger_long_fades_a_spike_below_the_lower_band() -> None:
    close = 100.0 + np.sin(np.arange(120) / 3.0)  # mild oscillation
    close[100] = 80.0  # crash through the lower band
    close[101] = 80.5  # still below the mean
    close[102] = 101.0  # back above the mean -> exit
    bars = _bars(close)
    train, test = bars.iloc[:90], bars.iloc[90:]
    sig = STRATEGIES["bollinger"].make_signal(train, test, {"window": 20, "num_std": 2.0}, "long")
    assert sig.loc[bars.index[100]] == 1.0
    assert sig.loc[bars.index[101]] == 1.0
    assert sig.loc[bars.index[102]] == 0.0


def test_bollinger_levels_limit_at_band_target_at_mean() -> None:
    rng = np.random.default_rng(7)
    bars = _bars(100.0 + np.cumsum(rng.normal(0, 0.5, 80)))
    lv = STRATEGIES["bollinger"].levels(bars, {"window": 20, "num_std": 2.0}, "long")
    mid = float(bars["close"].rolling(20).mean().iloc[-1])
    sd = float(bars["close"].rolling(20).std().iloc[-1])
    assert lv.entry == pytest.approx(mid - 2.0 * sd)
    assert lv.entry_type == "limit"
    assert lv.target == pytest.approx(mid)
    short = STRATEGIES["bollinger"].levels(bars, {"window": 20, "num_std": 2.0}, "short")
    assert short.entry == pytest.approx(mid + 2.0 * sd)


def test_registry_has_five_strategies_with_18_total_trials() -> None:
    assert set(STRATEGIES) == {"sma_cross", "donchian", "rsi2", "macd", "bollinger"}
    total = sum(len(expand_grid(s.param_grid)) for s in STRATEGIES.values())
    assert total == 18  # the global n_trials the tournament deflates by
    styles = {"trend", "breakout", "mean_reversion"}
    assert all(s.style in styles and s.description for s in STRATEGIES.values())
