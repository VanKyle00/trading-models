"""Tests for the per-ticker tournament engine (controlled registry for determinism)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradinglib.tournament import run as run_module
from tradinglib.tournament.levels import Levels
from tradinglib.tournament.run import TournamentConfig, run_tournament, survival_reasons
from tradinglib.tournament.strategies import STRATEGIES, StrategyDef

CONFIG = TournamentConfig(initial_train=100, test_size=50)


def _uptrend_bars(n: int = 600) -> pd.DataFrame:
    close = 100.0 * 1.01 ** np.arange(n)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _blocky_signal(train, test, params, stance):
    # long on alternating 5-bar blocks -> profitable on the uptrend AND many
    # round trips, so it clears both the DSR and the n_trades bars
    positions = (np.arange(len(test)) // 5) % 2
    return pd.Series(positions.astype(float), index=test.index)


def _flat_signal(train, test, params, stance):
    return pd.Series(0.0, index=test.index)


def _levels(bars, params, stance):
    return Levels(entry=1.0, entry_type="market", stop=0.5, target=2.0, condition="test")


def _def(key: str, signal, grid: dict) -> StrategyDef:
    return StrategyDef(
        key=key,
        name=key,
        style="trend",
        description=key,
        param_grid=grid,
        make_signal=signal,
        levels=_levels,
    )


REGISTRY = {
    "blocky": _def("blocky", _blocky_signal, {"a": [1, 2, 3]}),
    "flat": _def("flat", _flat_signal, {"b": [1, 2, 3, 4, 5]}),
}


def test_survival_reasons_each_criterion() -> None:
    cfg = TournamentConfig()
    good = {"deflated_sharpe": 0.95}
    assert survival_reasons(good, 20, {"a": 0.2}, cfg) == []
    assert "deflated_sharpe" in survival_reasons({"deflated_sharpe": 0.5}, 20, {}, cfg)[0]
    assert "n_trades" in survival_reasons(good, 5, {}, cfg)[0]
    assert "change-rate" in survival_reasons(good, 20, {"a": 0.8}, cfg)[0]
    assert len(survival_reasons({"deflated_sharpe": 0.0}, 0, {"a": 1.0}, cfg)) == 3


def test_run_tournament_picks_the_surviving_strategy() -> None:
    result = run_tournament(_uptrend_bars(), "long", registry=REGISTRY, config=CONFIG)
    assert result.n_trials == 8  # 3 + 5: the whole menu, not the winner's grid
    verdicts = {v.key: v for v in result.verdicts}
    assert verdicts["blocky"].survived
    assert not verdicts["flat"].survived
    assert verdicts["flat"].reasons  # every failed criterion is named
    assert result.winner is not None and result.winner.key == "blocky"
    assert result.winner_levels is not None  # levels built for the winner only
    assert type(verdicts["blocky"].params["a"]) is int  # JSON-native, not numpy


def test_run_tournament_deflates_by_the_global_trial_count(monkeypatch) -> None:
    seen: list[int] = []
    real = run_module.compute_metrics

    def spy(returns, equity_curve, periods_per_year=252, n_trials=1):
        seen.append(n_trials)
        return real(returns, equity_curve, periods_per_year=periods_per_year, n_trials=n_trials)

    monkeypatch.setattr(run_module, "compute_metrics", spy)
    run_tournament(_uptrend_bars(), "long", registry=REGISTRY, config=CONFIG)
    assert seen == [8, 8]  # both strategies re-scored against the full menu


def _always_long_signal(train, test, params, stance):
    # Holds a position through the entire OOS window: it never closes, so the
    # only "trade" is the dangling open one.
    return pd.Series(1.0, index=test.index)


def test_n_trades_excludes_unclosed_oos_position() -> None:
    # A strategy whose stitched OOS position never returns to flat has ZERO
    # completed round-trips; the dangling open position must not be counted
    # toward the min_trades survival gate (audit A3).
    registry = {"always": _def("always", _always_long_signal, {"a": [1]})}
    result = run_tournament(_uptrend_bars(), "long", registry=registry, config=CONFIG)
    assert result.verdicts[0].n_trades == 0


def _verdict(result, key: str):
    return next(v for v in result.verdicts if v.key == key)


def test_resting_engine_dark_by_default_and_routes_only_resting_strategies() -> None:
    # The flag ships dark: default off => the WHOLE tournament is byte-identical.
    # On => only strategies that carry a resting_orders builder change; a
    # position-only strategy stays bit-for-bit identical (audit A2 safety bar).
    bars = _uptrend_bars()
    reg = {"donchian": STRATEGIES["donchian"], "sma_cross": STRATEGIES["sma_cross"]}
    cfg_off = TournamentConfig(initial_train=100, test_size=50)
    cfg_on = TournamentConfig(initial_train=100, test_size=50, use_resting_engine=True)
    assert cfg_off.use_resting_engine is False  # dark by default

    off = run_tournament(bars, "long", registry=reg, config=cfg_off)
    on = run_tournament(bars, "long", registry=reg, config=cfg_on)

    # sma_cross has no resting_orders builder -> identical regardless of the flag
    assert _verdict(off, "sma_cross").as_dict() == _verdict(on, "sma_cross").as_dict()
    # donchian carries a builder -> the flag re-scores it on the resting engine
    assert _verdict(off, "donchian").as_dict() != _verdict(on, "donchian").as_dict()


def test_resting_path_uses_the_exact_completed_trade_count() -> None:
    # On the resting path the survival n_trades is the engine's exact round-trip
    # count (n_completed_trades), NOT trades_from_position (which merges adjacent
    # re-armed trades and would undercount — audit A2 must-fix #2).
    bars = _uptrend_bars()
    sdef = STRATEGIES["donchian"]
    cfg = TournamentConfig(initial_train=100, test_size=50, use_resting_engine=True)
    res = run_tournament(bars, "long", registry={"donchian": sdef}, config=cfg)

    wf = run_module.walk_forward(
        bars,
        make_orders=lambda tr, te, p: sdef.resting_orders(tr, te, p, "long"),
        stance="long",
        param_grid=sdef.param_grid,
        mode="anchored",
        initial_train=100,
        test_size=50,
        embargo=sdef.cv_embargo,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
    )
    assert wf.oos_result.n_completed_trades is not None
    assert _verdict(res, "donchian").n_trades == wf.oos_result.n_completed_trades


def test_registered_strategies_default_cv_embargo_to_zero() -> None:
    # Every shipped strategy is causal / self-purging, so its CV embargo is 0 —
    # purged CV is available but introduces NO change to current results until a
    # future strategy declares a forward-looking label horizon (audit A4).
    assert all(s.cv_embargo == 0 for s in STRATEGIES.values())


def test_run_tournament_applies_strategy_cv_embargo(monkeypatch) -> None:
    # A strategy that declares cv_embargo=N must have its walk-forward windows
    # purged by N bars — the CV layer enforces the purge, not the strategy.
    seen: list[int] = []
    real = run_module.walk_forward

    def spy(*args, **kwargs):
        seen.append(kwargs.get("embargo"))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "walk_forward", spy)
    registry = {
        "emb": StrategyDef(
            key="emb",
            name="emb",
            style="trend",
            description="emb",
            param_grid={"a": [1]},
            make_signal=_blocky_signal,
            levels=_levels,
            cv_embargo=4,
        )
    }
    run_tournament(_uptrend_bars(), "long", registry=registry, config=CONFIG)
    assert seen == [4]


def test_run_tournament_no_survivors_no_winner() -> None:
    result = run_tournament(
        _uptrend_bars(), "long", registry={"flat": REGISTRY["flat"]}, config=CONFIG
    )
    assert result.winner is None and result.winner_levels is None
    assert [v.survived for v in result.verdicts] == [False]


def test_run_tournament_insufficient_history_raises() -> None:
    with pytest.raises(ValueError, match="insufficient history"):
        run_tournament(_uptrend_bars(120), "long", registry=REGISTRY, config=CONFIG)


def test_run_tournament_real_registry_structural() -> None:
    bars = _uptrend_bars(700)
    result = run_tournament(bars, "long", config=TournamentConfig(initial_train=300, test_size=63))
    assert {v.key for v in result.verdicts} == set(STRATEGIES)
    assert result.n_trials == 29  # keep in sync with the registry (was 27; +2 for ridge_momentum)
    survivors = {v.key for v in result.verdicts if v.survived}
    if result.winner is None:
        assert not survivors
    else:
        assert result.winner.key in survivors
        assert result.winner_levels is not None
    for v in result.verdicts:
        assert v.survived == (not v.reasons)
        assert set(v.params) == set(STRATEGIES[v.key].param_grid)
        assert {"deflated_sharpe", "sharpe"} <= set(v.metrics)
