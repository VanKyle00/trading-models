"""03-earnings-straddle (SPY) — Phase-1 synthetic pipeline.

Long ATM straddle into earnings on a liquid name. The EDGE is a selection
filter: enter only when forecast realized move exceeds the implied move priced
into the straddle by a margin k (>1). Phase 1 prices a synthetic straddle off an
explicit pre-earnings IV and post-earnings crush (EventVolSurface) while the
realized move comes from real yfinance bars. NOT YET TRADEABLE — synthetic vol,
mirrors the repo's SP2 treatment. Phases 2 (free forward snapshots) and 3 (paid
chain history) are out of scope.

tz contract: load_daily returns a UTC-aware index; main()/run_for_gui strip tz
(tz_localize(None)) so the index matches the strategy's naive bar handling, and
run_synthetic normalizes earnings_datetime to tz-naive. Leakage: expected_move is
computed ONLY from prior earnings events (past_moves), never from the traded event.

Walk-forward across earnings seasons (Component 6, 4th method) is descoped this
cycle: validation/walk_forward.py is built on the vectorized run_backtest and is
incompatible with the OptionsEngine path; an options-aware walk-forward is a
separate design cycle. The Deflated-Sharpe n_trials hook in run_options_backtest
is the future wiring point for a parameter grid.

Outputs: results/metrics.json (SPY filtered branch), results/validation.json
(filtered vs unfiltered + bootstrap CI + cross-ticker FDR + trade metrics),
results/equity_curve.png.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest.metrics import (
    benjamini_hochberg_fdr,
    bootstrap_t_test,
    trade_metrics,
)
from tradinglib.backtest.options_engine import run_options_backtest
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.options.spread import ParametricSpread
from tradinglib.options.straddle import snap_strike, straddle_price
from tradinglib.options.surface import EventVolSurface

_HERE = Path(__file__).resolve().parent
SYMBOL = "SPY"
WATCHLIST = ["SPY", "AAPL", "MSFT", "AMZN", "NVDA"]
START = "2023-01-01"
END = "2024-12-31"
RATE = 0.04
FEE_BPS = 1.0
SLIPPAGE_BPS = 0.5
INITIAL_CAPITAL = 100_000.0
DEFAULT_K = 1.2
DEFAULT_LOOKBACK = 8
ENTRY_LEAD = 3
EXIT_OFFSET = 1
POST_TENOR = 14


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("es_strategy", _HERE / "strategy.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_signal_module():
    spec = importlib.util.spec_from_file_location("es_signal", _HERE / "signal.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_synthetic(
    *,
    close: pd.Series,
    earnings_datetime: pd.Timestamp,
    pre_iv: float,
    post_iv: float,
    k: float = DEFAULT_K,
    lookback: int = DEFAULT_LOOKBACK,
    past_moves: list[float] | None = None,
    prior_earnings: pd.Series | None = None,
) -> dict:
    """Run the synthetic straddle twice (filtered by the k-gate vs unfiltered).

    expected_move uses ONLY prior events: supply ``past_moves`` (realized abs
    moves of earlier events) or ``prior_earnings`` (dates strictly before the
    traded event); it never uses the traded event as its own history. Returns a
    dict with 'filtered'/'unfiltered' branches (metrics, final_equity, trade_pnl,
    took_trade) plus the implied/expected moves and k.
    """
    sig = _load_signal_module()
    strat = _load_strategy_module()

    cl = close.sort_index()
    if getattr(cl.index, "tz", None) is not None:
        cl = pd.Series(cl.to_numpy(), index=cl.index.tz_localize(None), name="close")
    ed = _to_naive(earnings_datetime)

    bars = list(cl.index)
    e_idx = next((i for i, b in enumerate(bars) if b >= ed), None)
    if e_idx is None or e_idx - ENTRY_LEAD < 0:
        raise ValueError(
            f"earnings {ed} out of range or too close to series start "
            f"(need entry_lead={ENTRY_LEAD} bars before the earnings bar)"
        )

    entry_spot = float(cl.iloc[e_idx - ENTRY_LEAD])
    strike = snap_strike(entry_spot, 1.0)
    t_years = max((ed + pd.Timedelta(days=POST_TENOR) - bars[e_idx - ENTRY_LEAD]).days, 1) / 365.0
    premium = straddle_price(entry_spot, strike, t_years, pre_iv, RATE)
    implied = sig.implied_move(premium, entry_spot)

    if past_moves is not None:
        vals = [abs(m) for m in past_moves][-lookback:]
        expected = float(pd.Series(vals).mean()) if vals else float("nan")
    elif prior_earnings is not None:
        prior = pd.to_datetime(prior_earnings, utc=True)
        prior = prior[prior < pd.Timestamp(earnings_datetime, tz="UTC")]
        expected = sig.expected_move(cl, prior, lookback)
    else:
        raise ValueError("supply past_moves or prior_earnings (no self-referential history)")

    surface = EventVolSurface(earnings_datetime=ed, pre_iv=pre_iv, post_iv=post_iv)

    def _branch(take: bool) -> dict:
        if not take:
            return {
                "took_trade": False,
                "metrics": {"sharpe": 0.0, "annualized_return": 0.0, "max_drawdown": 0.0},
                "final_equity": INITIAL_CAPITAL,
                "trade_pnl": 0.0,
            }
        s = strat.EarningsStraddle(
            earnings_datetime=ed,
            entry_lead=ENTRY_LEAD,
            exit_offset=EXIT_OFFSET,
            contracts=1.0,
            post_earnings_tenor=POST_TENOR,
            bar_index=cl.index,  # REQUIRED: Task 6 shipped a bar_index-based schedule
        )
        res = run_options_backtest(
            cl,
            s,
            surface=surface,
            spread=ParametricSpread(),
            rate=RATE,
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
            initial_capital=INITIAL_CAPITAL,
        )
        final_eq = float(res.equity_curve.iloc[-1])
        return {
            "took_trade": True,
            "metrics": res.metrics,
            "final_equity": final_eq,
            "trade_pnl": final_eq - INITIAL_CAPITAL,
            "implied_move": implied,
            "expected_move": expected,
        }

    return {
        "implied_move": implied,
        "expected_move": expected,
        "k": k,
        "filtered": _branch(sig.passes_filter(expected, implied, k)),
        "unfiltered": _branch(True),
    }


def build_validation_report(
    *,
    branches: dict[str, dict],
    per_ticker_pnl: dict[str, list[float]],
) -> dict:
    """Pure aggregation: pooled bootstrap CI, per-ticker FDR, pooled trade metrics.

    ``per_ticker_pnl[ticker]`` is the list of filtered per-event trade P&Ls for
    that ticker (n>=2 expected so the bootstrap is non-degenerate; a ticker with
    n<2 gets the sentinel p=1.0 and simply cannot be rejected).
    """
    pooled = pd.Series([p for ps in per_ticker_pnl.values() for p in ps], dtype=float)
    t_stat, ci_lo, ci_hi, p_value = bootstrap_t_test(pooled, n_boot=2000, seed=0)

    tickers = list(per_ticker_pnl.keys())
    pvals = [
        bootstrap_t_test(pd.Series(per_ticker_pnl[t]), n_boot=2000, seed=0)[3] for t in tickers
    ]
    rejected, fdr_threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)

    return {
        "phase": "1-synthetic-not-tradeable",
        "per_ticker": branches,
        "pooled_filtered": {
            "bootstrap_t_stat": t_stat,
            "bootstrap_ci_lower": ci_lo,
            "bootstrap_ci_upper": ci_hi,
            "bootstrap_p_value": p_value,
        },
        "fdr": {
            "tickers": tickers,
            "p_values": pvals,
            "rejected": rejected,
            "threshold": fdr_threshold,
        },
        "trade_metrics": trade_metrics(pooled),
    }


def plot_branches(branches: dict[str, dict], out_path: Path) -> None:
    """Bar chart of filtered vs unfiltered final equity per ticker."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(branches.keys())
    filt = [branches[n]["filtered"]["final_equity"] for n in names]
    unfilt = [branches[n]["unfiltered"]["final_equity"] for n in names]
    ax.bar([f"{n}\nfilt" for n in names], filt, label="filtered")
    ax.bar([f"{n}\nunfilt" for n in names], unfilt, label="unfiltered", alpha=0.6)
    ax.set_title("Earnings straddle: filtered vs unfiltered final equity (synthetic)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _events_for(ticker: str, close: pd.Series) -> pd.Series:
    """Real earnings dates within the price window (mocked-free path is data-optional)."""
    df = get_earnings_dates([ticker], start=START, end=END)
    return df["earnings_datetime"]


def _run_engine(close: pd.Series, earnings_datetime: pd.Timestamp, pre_iv: float, post_iv: float):
    """Run the straddle on one event and return its BacktestResult.

    ``earnings_datetime`` may be out of the data window (or too close to the
    start): the strategy plans no trade in that case and the engine returns a
    flat-equity result, so the GUI/service always gets a valid BacktestResult.
    """
    strat = _load_strategy_module()
    ed = _to_naive(earnings_datetime)
    surface = EventVolSurface(earnings_datetime=ed, pre_iv=pre_iv, post_iv=post_iv)
    s = strat.EarningsStraddle(
        earnings_datetime=ed,
        entry_lead=ENTRY_LEAD,
        exit_offset=EXIT_OFFSET,
        contracts=1.0,
        post_earnings_tenor=POST_TENOR,
        bar_index=close.index,
    )
    return run_options_backtest(
        close,
        s,
        surface=surface,
        spread=ParametricSpread(),
        rate=RATE,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        initial_capital=INITIAL_CAPITAL,
    )


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    k: float = DEFAULT_K,
    lookback: int = DEFAULT_LOOKBACK,
    entry_lead: int = ENTRY_LEAD,
    exit_offset: int = EXIT_OFFSET,
    pre_iv: float = 0.45,
    post_iv: float = 0.25,
) -> dict[str, Any]:
    """GUI / service adapter entry point.

    Runs the straddle on one illustrative event on ``symbol`` and returns the
    service contract (``data``, ``result``, ``symbol``, ``params``) plus the
    distinctive ``report`` (filtered vs unfiltered synthetic comparison) and a
    ``note`` set only when no earnings event exists in the window (e.g. SPY).
    The service stashes ``report``/``note`` in ``BacktestRun.extra``.

    entry_lead/exit_offset are accepted to match the model.md param schema (the
    synthetic phase uses the module-level ENTRY_LEAD/EXIT_OFFSET timing).
    """
    bars = load_daily(symbol, start=str(start), end=str(end))
    close = bars["close"]
    close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
    events = _events_for(symbol, close)
    events_naive = pd.to_datetime(events, utc=True).dt.tz_localize(None)
    usable = [
        e for e in events_naive if close.index[ENTRY_LEAD] <= e <= close.index[-EXIT_OFFSET - 1]
    ]

    report: dict | None = None
    note: str | None = None
    if usable:
        traded = usable[len(usable) // 2]
        prior = events[pd.to_datetime(events, utc=True) < pd.Timestamp(traded, tz="UTC")]
        report = run_synthetic(
            close=close,
            earnings_datetime=traded,
            pre_iv=pre_iv,
            post_iv=post_iv,
            k=k,
            lookback=lookback,
            prior_earnings=prior,
        )
    else:
        # No tradeable event in the window: plan against the (out-of-window) end
        # so the strategy simply never trades and the result is flat equity. Surface
        # a note so the resulting flat curve reads as intentional, not broken — the
        # default ticker (SPY) is an ETF with no earnings.
        traded = close.index[-1] + pd.Timedelta(days=365)
        note = (
            f"No earnings events found for {symbol} between {start} and {end}. "
            f"This model trades single-name equities around earnings; index ETFs "
            f"like SPY have none. Try a single name such as AAPL, MSFT, or NVDA, "
            f"or widen the date range."
        )

    result = _run_engine(close, traded, pre_iv, post_iv)
    data = pd.DataFrame({"close": close, "position": result.position})
    return {
        "data": data,
        "result": result,
        "report": report,
        "note": note,
        "symbol": symbol,
        "params": {
            "start": str(start),
            "end": str(end),
            "symbol": symbol,
            "k": k,
            "lookback": lookback,
            "entry_lead": entry_lead,
            "exit_offset": exit_offset,
            "pre_iv": pre_iv,
            "post_iv": post_iv,
        },
    }


def main() -> None:
    out_dir = _HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    branches: dict[str, dict] = {}
    per_ticker_pnl: dict[str, list[float]] = {}
    for ticker in WATCHLIST:
        try:
            bars = load_daily(ticker, START, END)
        except Exception:
            continue
        close = bars["close"]
        close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
        try:
            events = _events_for(ticker, close)
        except Exception:
            continue
        events_naive = sorted(pd.to_datetime(events, utc=True).dt.tz_localize(None))
        usable = [
            e for e in events_naive if close.index[ENTRY_LEAD] <= e <= close.index[-EXIT_OFFSET - 1]
        ]
        pnl: list[float] = []
        last_report: dict | None = None
        for traded in usable:
            prior = events[pd.to_datetime(events, utc=True) < pd.Timestamp(traded, tz="UTC")]
            try:
                rep = run_synthetic(
                    close=close,
                    earnings_datetime=traded,
                    pre_iv=0.45,
                    post_iv=0.25,
                    prior_earnings=prior,
                )
            except ValueError:
                continue
            last_report = rep
            f = rep["filtered"]
            # A non-fired k-gate is NO trade, not a flat $0 trade: skip sit-out
            # events entirely. per_ticker_pnl is fed verbatim to bootstrap_t_test
            # and trade_metrics, which drop only NaN (not zeros), so zero-padding
            # sit-outs would overcount n_trades and dilute win_rate/expectancy.
            if f["took_trade"]:
                pnl.append(float(f["trade_pnl"]))
        if last_report is not None and pnl:
            branches[ticker] = last_report
            per_ticker_pnl[ticker] = pnl

    if not branches:
        print(json.dumps({"phase": "1-synthetic-not-tradeable", "note": "no data"}, indent=2))
        return

    validation = build_validation_report(branches=branches, per_ticker_pnl=per_ticker_pnl)
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=str))

    # Headline metrics.json keys off the default ticker (SPY) when it traded;
    # otherwise fall back to the first ticker whose k-gate fired, so the file is
    # never silently empty just because the default sat out. Per-ticker detail
    # lives in validation.json. branches is non-empty here (guarded above).
    headline_ticker = "SPY" if "SPY" in branches else next(iter(branches))
    headline = branches[headline_ticker].get("filtered", {}).get("metrics", {})
    (out_dir / "metrics.json").write_text(json.dumps(headline, indent=2, default=str))
    plot_branches(branches, out_dir / "equity_curve.png")
    print(json.dumps(validation, indent=2, default=str))


if __name__ == "__main__":
    main()
