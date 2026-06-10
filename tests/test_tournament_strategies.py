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
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01 if high is None else np.asarray(high, dtype=float),
            "low": close * 0.99 if low is None else np.asarray(low, dtype=float),
            "close": close,
        },
        index=idx,
    )
    bars["volume"] = np.full(len(close), 1_000_000.0)
    return bars


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


def test_donchian_degenerate_channel_returns_none() -> None:
    # A dead-flat channel has no stop distance; the contract answer is
    # None ("no actionable entry"), not an exception that poisons the ticker.
    bars = _flat_bars(n=80, spread=0.0)  # high == low == close
    lv = STRATEGIES["donchian"].levels(bars, {"n": 20}, "long")
    assert lv is None


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


def test_macd_long_in_uptrend_short_on_breakdown() -> None:
    params = {"fast": 12, "slow": 26, "signal": 9}
    up = _trend_bars()
    sig = STRATEGIES["macd"].make_signal(up.iloc[:300], up.iloc[300:], params, "long")
    assert set(np.unique(sig)) <= {0.0, 1.0}
    assert sig.iloc[-1] == 1.0
    # A constant-rate decay converges to MACD >= signal (the signal line lags a
    # shrinking-magnitude negative), so the short rule needs WORSENING momentum:
    # flat then breaking down keeps the MACD line under its own lagging EMA.
    breakdown = _bars(np.concatenate([np.full(300, 100.0), 100.0 * 0.99 ** np.arange(1, 31)]))
    train, test = breakdown.iloc[:300], breakdown.iloc[300:]
    short = STRATEGIES["macd"].make_signal(train, test, params, "short")
    assert set(np.unique(short)) <= {-1.0, 0.0}
    assert (short == -1.0).all()  # MACD below its signal line on every decline bar


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


def test_registry_has_nine_strategies_with_29_total_trials() -> None:
    assert set(STRATEGIES) == {
        "sma_cross",
        "donchian",
        "rsi2",
        "macd",
        "bollinger",
        "base_breakout",
        "ma_pullback",
        "pead",
        "ridge_momentum",
    }  # keep in sync with the registry
    total = sum(len(expand_grid(s.param_grid)) for s in STRATEGIES.values())
    assert total == 29  # keep in sync with the registry (was 27; +2 for ridge_momentum)
    styles = {"trend", "breakout", "mean_reversion", "event", "ml"}
    assert all(s.style in styles and s.description for s in STRATEGIES.values())


def _base_breakout_bars(n_base: int = 60) -> pd.DataFrame:
    # 252 bars of uptrend to a high, then a tight drying-volume base just
    # under it, then a breakout bar above the base high.
    trend = 100.0 * 1.004 ** np.arange(252)
    base = np.full(n_base, trend[-1] * 0.99)
    breakout = np.array([trend[-1] * 1.03])
    close = np.concatenate([trend, base, breakout])
    bars = _bars(close, high=close * 1.002, low=close * 0.998)
    vol = np.full(len(close), 1_000_000.0)
    # volume must DECLINE through the base: a constant low value makes the
    # trailing-50 ratio converge to 1.0 late in the base and the dry-volume
    # gate (ratio < 1.0) would fail right before the breakout
    vol[-(n_base + 1) : -1] = np.linspace(900_000.0, 300_000.0, n_base)
    bars["volume"] = vol
    return bars


def test_base_breakout_long_fires_on_breakout_from_tight_base() -> None:
    bars = _base_breakout_bars()
    train, test = bars.iloc[:280], bars.iloc[280:]
    sig = STRATEGIES["base_breakout"].make_signal(train, test, {"base": 40}, "long")
    assert sig.iloc[-1] == 1.0  # long on the breakout bar
    assert set(np.unique(sig)) <= {0.0, 1.0}


def test_base_breakout_levels_actionable_only_in_a_base() -> None:
    bars = _base_breakout_bars()
    lv = STRATEGIES["base_breakout"].levels(bars.iloc[:-1], {"base": 40}, "long")
    assert lv is not None and lv.entry_type == "stop"
    assert lv.entry == pytest.approx(float(bars["high"].iloc[:-1].iloc[-40:].max()))
    assert lv.stop < lv.entry < lv.target
    # an established uptrend with no consolidation is not actionable
    trending = _trend_bars(n=420)
    assert STRATEGIES["base_breakout"].levels(trending, {"base": 40}, "long") is None


def _ma_pullback_bars() -> pd.DataFrame:
    # long uptrend, then a controlled pullback onto the rising SMA50:
    # 10 down bars at 0.8%/bar stall the trend enough, 9 up bars at 0.3%/bar
    # park the close inside the SMA50 ±3% band with RSI(14) ≈ 40 in [30, 50].
    up = 100.0 * 1.003 ** np.arange(420)
    seg = [up[-1]]
    for _ in range(10):
        seg.append(seg[-1] * 0.992)
    for _ in range(9):
        seg.append(seg[-1] * 1.003)
    dip = np.array(seg[1:])
    return _bars(np.concatenate([up, dip]))


def test_ma_pullback_levels_actionable_at_the_ma() -> None:
    bars = _ma_pullback_bars()
    lv = STRATEGIES["ma_pullback"].levels(bars, {"band": 0.03}, "long")
    assert lv is not None  # fixture was tuned so RSI(14) ≈ 40 lands in [30, 50]
    assert lv.entry_type == "stop"
    assert lv.stop < lv.entry < lv.target
    # no pullback -> never actionable
    assert STRATEGIES["ma_pullback"].levels(_trend_bars(), {"band": 0.03}, "long") is None


def test_ma_pullback_signal_is_zero_in_flat_chop() -> None:
    bars = _flat_bars(n=420)
    sig = STRATEGIES["ma_pullback"].make_signal(
        bars.iloc[:300], bars.iloc[300:], {"band": 0.03}, "long"
    )
    assert (sig == 0.0).all()


def test_ma_pullback_signal_enters_on_sma20_recapture_not_in_the_dip() -> None:
    # Fixture probe: base ends at bar 438 with setup=True on the last two bars
    # (armed=True on bar 438). rec_bar 0 closes above SMA20 with armed=True so
    # the entry fires immediately on the first recovery bar regardless of rate.
    base = _ma_pullback_bars()  # ends inside the dip (close < SMA20)
    recovery = base["close"].iloc[-1] * 1.012 ** np.arange(1, 9)
    close = np.concatenate([base["close"].to_numpy(), recovery])
    bars = _bars(close)
    train, test = bars.iloc[:300], bars.iloc[300:]
    sig = STRATEGIES["ma_pullback"].make_signal(train, test, {"band": 0.03}, "long")
    in_dip = sig.loc[base.index[-3] : base.index[-1]]
    assert (in_dip == 0.0).all()  # no position while still below SMA(20)
    assert sig.iloc[-1] == 1.0  # long after the recapture bar


def test_base_breakout_short_fires_on_breakdown_from_tight_base(n_base: int = 60) -> None:
    trend = 100.0 * 0.996 ** np.arange(252)  # downtrend to the 52w low
    base = np.full(n_base, trend[-1] * 1.01)
    breakdown = np.array([trend[-1] * 0.97])
    close = np.concatenate([trend, base, breakdown])
    bars = _bars(close, high=close * 1.002, low=close * 0.998)
    vol = np.full(len(close), 1_000_000.0)
    vol[-(n_base + 1) : -1] = np.linspace(900_000.0, 300_000.0, n_base)
    bars["volume"] = vol
    train, test = bars.iloc[:280], bars.iloc[280:]
    sig = STRATEGIES["base_breakout"].make_signal(train, test, {"base": 40}, "short")
    assert sig.iloc[-1] == -1.0
    lv = STRATEGIES["base_breakout"].levels(bars.iloc[:-1], {"base": 40}, "short")
    assert lv is not None and lv.target < lv.entry < lv.stop


def _pead_bars() -> pd.DataFrame:
    # flat, then a +8% earnings gap on 3x volume, then upward drift
    pre = np.full(80, 100.0)
    post = 108.0 * 1.002 ** np.arange(20)
    close = np.concatenate([pre, [108.0], post])
    bars = _bars(close)
    vol = np.full(len(close), 1_000_000.0)
    vol[80] = 3_000_000.0
    bars["volume"] = vol
    earnings = np.zeros(len(close), dtype=bool)
    earnings[80] = True
    bars["earnings"] = earnings
    return bars


def test_pead_long_drifts_after_the_gap() -> None:
    bars = _pead_bars()
    train, test = bars.iloc[:60], bars.iloc[60:]
    sig = STRATEGIES["pead"].make_signal(train, test, {"min_move": 0.04, "hold": 12}, "long")
    assert sig.iloc[-10] == 1.0  # in the drift window
    assert sig.iloc[0] == 0.0  # not before the reaction


def test_pead_without_earnings_column_is_flat_and_unactionable() -> None:
    bars = _trend_bars(n=200)
    train, test = bars.iloc[:150], bars.iloc[150:]
    sig = STRATEGIES["pead"].make_signal(train, test, {"min_move": 0.04, "hold": 12}, "long")
    assert (sig == 0.0).all()
    assert STRATEGIES["pead"].levels(bars, {"min_move": 0.04, "hold": 12}, "long") is None


def test_pead_levels_inside_the_drift_window() -> None:
    bars = _pead_bars()
    lv = STRATEGIES["pead"].levels(bars.iloc[:86], {"min_move": 0.04, "hold": 12}, "long")
    assert lv is not None and lv.entry_type == "market"
    assert lv.stop == pytest.approx(float(bars["low"].iloc[80]))


def test_pead_requires_both_move_and_volume_gates() -> None:
    # +8% earnings move on NORMAL volume -> no reaction, no drift position
    quiet = _pead_bars()
    quiet.loc[quiet.index[80], "volume"] = 1_000_000.0
    sig = STRATEGIES["pead"].make_signal(
        quiet.iloc[:60], quiet.iloc[60:], {"min_move": 0.04, "hold": 12}, "long"
    )
    assert (sig == 0.0).all()
    # 3x volume on a sub-threshold move -> equally flat
    small = _pead_bars()
    small_close = small["close"].to_numpy().copy()
    small_close[80:] = small_close[79] * 1.01  # +1% < 4% min_move
    small["close"] = small_close
    sig2 = STRATEGIES["pead"].make_signal(
        small.iloc[:60], small.iloc[60:], {"min_move": 0.04, "hold": 12}, "long"
    )
    assert (sig2 == 0.0).all()


def test_ridge_momentum_long_rides_a_persistent_trend() -> None:
    # ML output is fit-dependent and not hand-predictable bar-by-bar, so the
    # assertion is statistical: in a strong steady uptrend the momentum model
    # should be long most of the time, not on any specific bar.
    bars = _trend_bars(n=700, rate=1.003)
    train, test = bars.iloc[:500], bars.iloc[500:]
    sig = STRATEGIES["ridge_momentum"].make_signal(train, test, {"l2": 1.0}, "long")
    assert set(np.unique(sig)) <= {0.0, 1.0}
    assert float(sig.iloc[-100:].mean()) > 0.5


def test_ridge_momentum_flat_market_stays_flat() -> None:
    bars = _flat_bars(n=420)
    sig = STRATEGIES["ridge_momentum"].make_signal(
        bars.iloc[:300], bars.iloc[300:], {"l2": 1.0}, "long"
    )
    assert (sig == 0.0).all()


def test_ridge_momentum_levels_market_entry_in_trend_none_when_flat() -> None:
    bars = _trend_bars(n=700, rate=1.003)
    lv = STRATEGIES["ridge_momentum"].levels(bars, {"l2": 1.0}, "long")
    assert lv is not None and lv.entry_type == "market"
    assert lv.stop < lv.entry < lv.target
    assert STRATEGIES["ridge_momentum"].levels(_flat_bars(n=420), {"l2": 1.0}, "long") is None
