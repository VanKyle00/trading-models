"""Intrabar resting-order scoring for the tournament (audit A2).

:func:`run_resting_backtest` scores a strategy that issues a RESTING stop/limit
order — filled intrabar on touch at the level — instead of a ``{0, ±1}`` position
series filled at the next open. It fills via the SAME primitives the forward
ledger uses (:mod:`tradinglib.backtest.fills`, shared with ``simulate_ticket``),
so the certified backtest validates the trade the issued ticket actually takes
(cert==ticket).

The ``orders`` frame is the per-bar resting order in force GOING INTO each bar,
with columns ``entry, stop, target, entry_type, armed``. Its levels MUST already
be causal — computed from data ``<= t-1`` by the caller (e.g.
``high.rolling(n).max().shift(1)``); the fill on bar ``t`` then reads bar ``t``'s
own OHLC, exactly as ``simulate_ticket`` does. The process is continuous: after an
exit the order re-arms on the NEXT bar (a position never exits and re-enters on
the same bar), so each completed round-trip is counted independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglib.backtest.engine import BacktestResult
from tradinglib.backtest.fills import _entry_fill, _exit_fill
from tradinglib.backtest.metrics import compute_metrics

_REQUIRED_COLUMNS = ("entry", "stop", "target", "entry_type", "armed")


def _is_num(x: object) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x))


def simulate_resting(bars: pd.DataFrame, orders: pd.DataFrame, stance: str) -> list[dict]:
    """Walk the bars, re-arming the resting order after each exit.

    Returns the trades in chronological order. Each is a dict with
    ``entry_idx/entry_fill/stop/target/risk`` and ``exit_idx/exit_fill/status/
    ambiguous`` (``exit_idx`` is ``None`` and ``status`` ``"open"`` for a trade
    still open at the slice end). Fill conventions are ``fills._entry_fill`` /
    ``fills._exit_fill`` verbatim, including both-touched=>stopped and the
    limit-entry-bar target suppression, so they match ``simulate_ticket``.
    """
    n = len(bars)
    trades: list[dict] = []
    i = 0
    while i < n:
        o = orders.iloc[i]
        entry, stop, target = o["entry"], o["stop"], o["target"]
        if not (bool(o["armed"]) and _is_num(entry) and _is_num(stop) and _is_num(target)):
            i += 1
            continue
        entry, stop, target = float(entry), float(stop), float(target)
        risk = abs(entry - stop)
        if risk == 0.0:
            i += 1
            continue
        fill = _entry_fill(bars.iloc[i], entry, o["entry_type"], stance)
        if fill is None:
            i += 1
            continue
        is_limit = o["entry_type"] == "limit"
        trade = {
            "entry_idx": i,
            "entry_fill": fill,
            "stop": stop,
            "target": target,
            "risk": risk,
            "exit_idx": None,
            "exit_fill": None,
            "status": "open",
            "ambiguous": False,
        }
        for j in range(i, n):
            exit_fill, status, ambiguous = _exit_fill(bars.iloc[j], stop, target, stance)
            if exit_fill is None:
                continue
            # On the limit-entry bar a target-only touch does NOT exit (intrabar
            # order is unknowable); a stop hit still exits (matches simulate_ticket).
            if j == i and is_limit and status == "target":
                continue
            trade.update(exit_idx=j, exit_fill=exit_fill, status=status, ambiguous=ambiguous)
            break
        trades.append(trade)
        if trade["exit_idx"] is None:
            break  # open at the slice end; nothing left to re-arm
        i = trade["exit_idx"] + 1  # re-arm strictly after the exit bar
    return trades


def run_resting_backtest(
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    *,
    stance: str,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    periods_per_year: int = 252,
    n_trials: int = 1,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Score a resting-order strategy and return a standard :class:`BacktestResult`.

    ``returns`` is a per-bar net-return series on the same price-fraction basis as
    the position-series engine (entry bar: entry_fill->close; held bars:
    close-to-close; exit bar: prev_close->exit_fill; a single-bar round-trip:
    entry_fill->exit_fill), so the Deflated Sharpe is comparable across strategies.
    ``n_completed_trades`` is the exact closed round-trip count (the survival
    gate's trade count), not re-derived from the merge-prone position series.
    """
    if not orders.index.equals(bars.index):
        raise ValueError("orders and bars must share the same index")
    missing = [c for c in _REQUIRED_COLUMNS if c not in orders]
    if missing:
        raise ValueError(f"orders frame missing columns: {missing}")

    sign = 1.0 if stance == "long" else -1.0
    closes = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    gross = np.zeros(n)
    position = np.zeros(n)
    completed = 0

    for t in simulate_resting(bars, orders, stance):
        e = t["entry_idx"]
        x = t["exit_idx"]
        last = x if x is not None else n - 1
        if x is not None:
            completed += 1
        position[e : last + 1] = sign
        ef = t["entry_fill"]
        if last == e:
            ref = t["exit_fill"] if x is not None else closes[e]
            gross[e] = sign * (ref - ef) / ef
        else:
            gross[e] = sign * (closes[e] - ef) / ef
            for k in range(e + 1, last):
                gross[k] = sign * (closes[k] - closes[k - 1]) / closes[k - 1]
            ref = t["exit_fill"] if x is not None else closes[last]
            gross[last] = sign * (ref - closes[last - 1]) / closes[last - 1]

    position_s = pd.Series(position, index=bars.index)
    turnover = position_s.diff().abs()
    turnover.iloc[0] = abs(position[0])
    cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    net = pd.Series(gross, index=bars.index) - cost
    equity = (1.0 + net).cumprod() * initial_capital
    metrics = compute_metrics(net, equity, periods_per_year=periods_per_year, n_trials=n_trials)
    return BacktestResult(
        equity_curve=equity,
        returns=net,
        position=position_s,
        turnover=turnover,
        metrics=metrics,
        config={
            "initial_capital": initial_capital,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "periods_per_year": periods_per_year,
            "n_trials": n_trials,
            "execution": "resting_intrabar",
        },
        n_completed_trades=completed,
    )
