"""Tests for the vectorized backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.backtest import BacktestResult, run_backtest


@pytest.fixture
def flat_prices() -> pd.Series:
    """100 bars of unchanging price."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(100.0, index=idx)


@pytest.fixture
def rising_prices() -> pd.Series:
    """Geometric 1%-per-bar uptrend across 100 bars."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = 100.0 * (1.01 ** np.arange(100))
    return pd.Series(prices, index=idx)


def test_returns_result_object(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fee_bps=0, slippage_bps=0)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(rising_prices)
    assert len(result.returns) == len(rising_prices)


def test_long_position_captures_uptrend(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fee_bps=0, slippage_bps=0)
    # Long full size on a 1%/bar uptrend with no costs — equity must grow.
    assert result.equity_curve.iloc[-1] > 150_000.0


def test_zero_position_means_flat(rising_prices: pd.Series) -> None:
    signals = pd.Series(0.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fee_bps=0, slippage_bps=0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_short_position_inverts_uptrend(rising_prices: pd.Series) -> None:
    signals = pd.Series(-1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fee_bps=0, slippage_bps=0)
    assert result.equity_curve.iloc[-1] < 100_000.0


def test_flat_prices_yield_flat_equity(flat_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=flat_prices.index)
    result = run_backtest(flat_prices, signals, fee_bps=0, slippage_bps=0)
    # No price change → no return; one unit of turnover at bar 1 brings tiny
    # cost only if fees > 0 (they're 0 here).
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_costs_drag_on_active_strategy(rising_prices: pd.Series) -> None:
    flipping = pd.Series(
        np.where(np.arange(len(rising_prices)) % 2 == 0, 1.0, -1.0),
        index=rising_prices.index,
    )
    no_cost = run_backtest(rising_prices, flipping, fee_bps=0, slippage_bps=0)
    with_cost = run_backtest(rising_prices, flipping, fee_bps=10, slippage_bps=5)
    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]


def test_signal_is_lagged_no_lookahead(flat_prices: pd.Series) -> None:
    # If we somehow used the same-bar signal, a price-aware signal could
    # capture the move within its own bar. Here we feed a signal that 'knows'
    # the future on a flat series — equity should still not move.
    signals = pd.Series(1.0, index=flat_prices.index)
    result = run_backtest(flat_prices, signals, fee_bps=0, slippage_bps=0)
    assert result.position.iloc[0] == 0.0  # first bar's position is 0 (lagged)


def test_mismatched_index_raises() -> None:
    a = pd.Series([100.0, 101.0], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    b = pd.Series([1.0, 1.0], index=pd.date_range("2021-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError, match="same index"):
        run_backtest(a, b)


def test_short_series_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    prices = pd.Series([100.0], index=idx)
    signals = pd.Series([1.0], index=idx)
    with pytest.raises(ValueError, match="at least 2"):
        run_backtest(prices, signals)


def test_metrics_keys_present(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals)
    for key in (
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hit_rate",
        "n_bars",
    ):
        assert key in result.metrics


def test_max_drawdown_is_nonpositive(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fee_bps=0, slippage_bps=0)
    assert result.metrics["max_drawdown"] <= 0.0


def test_n_trials_lowers_deflated_sharpe(rising_prices: pd.Series) -> None:
    # A noisy signal so returns have non-zero variance and a positive mean.
    rng = np.random.default_rng(3)
    signals = pd.Series(
        rng.integers(0, 2, size=len(rising_prices)).astype(float), index=rising_prices.index
    )
    one = run_backtest(rising_prices, signals, n_trials=1)
    many = run_backtest(rising_prices, signals, n_trials=50)
    assert many.metrics["deflated_sharpe"] <= one.metrics["deflated_sharpe"]
    assert "n_trials" in many.config and many.config["n_trials"] == 50


def test_execution_prices_use_open_on_entry_bar() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 110.0, 121.0], index=idx)
    opens = pd.Series([100.0, 105.0, 121.0], index=idx)
    signals = pd.Series([1.0, 1.0, 1.0], index=idx)

    base = run_backtest(close, signals, fee_bps=0, slippage_bps=0)
    nextopen = run_backtest(close, signals, execution_prices=opens, fee_bps=0, slippage_bps=0)

    # Entry happens at bar 1 (position 0->1). Next-open fills earn close/open-1
    # on that bar instead of the full close-to-close move.
    assert nextopen.returns.iloc[1] == pytest.approx(110.0 / 105.0 - 1.0)
    assert base.returns.iloc[1] == pytest.approx(110.0 / 100.0 - 1.0)
    # Held bar 2 (no position change) is close-to-close in both.
    assert nextopen.returns.iloc[2] == pytest.approx(121.0 / 110.0 - 1.0)
    # Next-open is strictly more conservative here.
    assert nextopen.equity_curve.iloc[-1] < base.equity_curve.iloc[-1]


def test_execution_prices_none_is_unchanged(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    a = run_backtest(rising_prices, signals, fee_bps=1.0, slippage_bps=0.5)
    b = run_backtest(rising_prices, signals, execution_prices=None, fee_bps=1.0, slippage_bps=0.5)
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
    pd.testing.assert_series_equal(a.returns, b.returns)


def test_execution_prices_index_mismatch_raises(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    bad_opens = pd.Series(
        1.0, index=pd.date_range("2099-01-01", periods=len(rising_prices), freq="D")
    )
    with pytest.raises(ValueError, match="execution_prices"):
        run_backtest(rising_prices, signals, execution_prices=bad_opens)


def test_next_open_is_default_and_requires_opens(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    with pytest.raises(ValueError, match="open_prices"):
        run_backtest(rising_prices, signals)


def test_decision_close_matches_legacy_closetoclose(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    result = run_backtest(rising_prices, signals, fill="decision_close", fee_bps=0, slippage_bps=0)
    assert result.equity_curve.iloc[-1] > 150_000.0
    assert result.config["execution"] == "decision_close"


def test_open_prices_next_open_entry_bar() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 110.0, 121.0], index=idx)
    opens = pd.Series([100.0, 105.0, 121.0], index=idx)
    signals = pd.Series([1.0, 1.0, 1.0], index=idx)
    res = run_backtest(close, signals, open_prices=opens, fee_bps=0, slippage_bps=0)
    assert res.returns.iloc[1] == pytest.approx(110.0 / 105.0 - 1.0)
    assert res.config["execution"] == "next_open"


def test_execution_prices_alias_warns_and_matches() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 110.0, 121.0], index=idx)
    opens = pd.Series([100.0, 105.0, 121.0], index=idx)
    signals = pd.Series([1.0, 1.0, 1.0], index=idx)
    with pytest.warns(DeprecationWarning, match="execution_prices"):
        legacy = run_backtest(close, signals, execution_prices=opens, fee_bps=0, slippage_bps=0)
    explicit = run_backtest(close, signals, open_prices=opens, fee_bps=0, slippage_bps=0)
    pd.testing.assert_series_equal(legacy.equity_curve, explicit.equity_curve)


def test_open_prices_index_mismatch_raises(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    bad = pd.Series(1.0, index=pd.date_range("2099-01-01", periods=len(rising_prices), freq="D"))
    with pytest.raises(ValueError, match="open_prices"):
        run_backtest(rising_prices, signals, open_prices=bad)


def test_open_prices_and_execution_prices_both_raises(rising_prices: pd.Series) -> None:
    signals = pd.Series(1.0, index=rising_prices.index)
    opens = rising_prices.copy()
    with pytest.raises(ValueError, match="not both"):
        run_backtest(rising_prices, signals, open_prices=opens, execution_prices=opens)
