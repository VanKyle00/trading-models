"""Per-ticker tournament: walk-forward every registered strategy, keep survivors.

Every strategy is anchored-walk-forward tested on the ticker's bars; the
stitched OOS slice is then re-scored with ``n_trials`` = the sum of ALL
registered grids — the Deflated Sharpe penalty for the whole menu we tried,
not just the winning strategy's own grid (CON-02 multiple-testing
discipline). Survival is a hard bar (DSR probability, trade count, parameter
stability); most tickers should have zero survivors most nights, and that is
the correct output, not a failure.  A winner whose ``levels`` builder returns
``None`` has no actionable entry tonight; the pipeline records the survivor
but issues no ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradinglib.backtest.metrics import compute_metrics
from tradinglib.eval.trades import trades_from_position
from tradinglib.tournament.levels import Levels
from tradinglib.tournament.strategies import STRATEGIES, StrategyDef
from tradinglib.validation.search import expand_grid
from tradinglib.validation.walk_forward import walk_forward


@dataclass(frozen=True)
class TournamentConfig:
    """Walk-forward sizing and survival thresholds (defaults from the design spec)."""

    initial_train: int = 378  # ~18 months of daily bars
    test_size: int = 63  # ~1 quarter; step == test_size (non-overlapping OOS)
    min_trades: int = 12
    dsr_threshold: float = 0.90
    max_param_change_rate: float = 0.5
    fee_bps: float = 1.0
    slippage_bps: float = 0.5


@dataclass
class StrategyVerdict:
    key: str
    survived: bool
    reasons: list[str]  # failed criteria; empty when survived
    params: dict  # the last window's selection — tomorrow's parameters
    metrics: dict  # stitched-OOS metrics, deflated by the global trial count
    n_trades: int
    param_stability: dict
    n_windows: int

    def as_dict(self) -> dict:
        return {
            "strategy": self.key,
            "survived": self.survived,
            "reasons": self.reasons,
            "params": self.params,
            "deflated_sharpe": self.metrics["deflated_sharpe"],
            "sharpe": self.metrics["sharpe"],
            "n_trades": self.n_trades,
            "max_param_change_rate": max(self.param_stability.values(), default=0.0),
            "n_windows": self.n_windows,
        }


@dataclass
class TournamentResult:
    stance: str
    verdicts: list[StrategyVerdict] = field(default_factory=list)
    winner: StrategyVerdict | None = None
    winner_levels: Levels | None = None
    n_trials: int = 0


def survival_reasons(
    metrics: dict, n_trades: int, param_stability: dict, config: TournamentConfig
) -> list[str]:
    """The survival bar — names every failed criterion, not just the first."""
    reasons = []
    dsr = metrics["deflated_sharpe"]
    if dsr < config.dsr_threshold:
        reasons.append(f"deflated_sharpe {dsr:.2f} < {config.dsr_threshold:.2f}")
    if n_trades < config.min_trades:
        reasons.append(f"n_trades {n_trades} < {config.min_trades}")
    change_rate = max(param_stability.values(), default=0.0)
    if change_rate > config.max_param_change_rate:
        reasons.append(f"param change-rate {change_rate:.2f} > {config.max_param_change_rate:.2f}")
    return reasons


def _native(value: object) -> object:
    """Unbox numpy scalars from DataFrame rows so reports stay JSON-native."""
    return value.item() if hasattr(value, "item") else value


def run_tournament(
    bars: pd.DataFrame,
    stance: str,
    registry: dict[str, StrategyDef] | None = None,
    config: TournamentConfig | None = None,
) -> TournamentResult:
    """Walk-forward every strategy in ``registry`` on one ticker's bars."""
    registry = STRATEGIES if registry is None else registry
    config = TournamentConfig() if config is None else config
    if not registry:
        raise ValueError("empty strategy registry")
    min_bars = config.initial_train + config.test_size
    if len(bars) < min_bars:
        raise ValueError(f"insufficient history: {len(bars)} bars < {min_bars} required")

    n_trials = sum(len(expand_grid(s.param_grid)) for s in registry.values())
    result = TournamentResult(stance=stance, n_trials=n_trials)

    for sdef in registry.values():

        def signal_fn(
            train: pd.DataFrame,
            test: pd.DataFrame,
            params: dict,
            _sdef: StrategyDef = sdef,
        ) -> pd.Series:
            return _sdef.make_signal(train, test, params, stance)

        wf = walk_forward(
            bars,
            signal_fn,
            param_grid=sdef.param_grid,
            mode="anchored",
            initial_train=config.initial_train,
            test_size=config.test_size,
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
        )
        # Re-score the stitched OOS slice deflated by the WHOLE menu's trial
        # count — walk_forward's own deflation only knows this strategy's grid.
        metrics = compute_metrics(
            wf.oos_result.returns, wf.oos_result.equity_curve, n_trials=n_trials
        )
        n_trades = len(trades_from_position(wf.oos_result.position, bars["close"]))
        last = wf.windows.iloc[-1]
        params = {k: _native(last[f"param_{k}"]) for k in sdef.param_grid}
        reasons = survival_reasons(metrics, n_trades, wf.param_stability, config)
        result.verdicts.append(
            StrategyVerdict(
                key=sdef.key,
                survived=not reasons,
                reasons=reasons,
                params=params,
                metrics=metrics,
                n_trades=n_trades,
                param_stability=wf.param_stability,
                n_windows=len(wf.windows),
            )
        )

    survivors = [v for v in result.verdicts if v.survived]
    if survivors:
        result.winner = sorted(survivors, key=lambda v: (-v.metrics["deflated_sharpe"], v.key))[0]
        result.winner_levels = registry[result.winner.key].levels(
            bars, result.winner.params, stance
        )
    return result
