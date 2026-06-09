"""Thorough backtest of model 03 (earnings event-vol straddle).

Reuses the model's OWN functions (run_synthetic, build_validation_report) so the
numbers are faithful to the shipped pipeline, but runs them over a much larger
universe and window than the model's default 2023-2024 / 5-name watchlist:

  * universe: 9 liquid optionable single names (SPY excluded - no earnings)
  * window:   each name's full cached earnings history (~2020-2026, ~24 events)
  * baseline: model defaults (k=1.2, lookback=8, pre_iv=0.45, post_iv=0.25)
  * sensitivities: pre_iv level (the decisive check - it shows the result's sign
    is an artifact of this one assumed input, swinging PF 4.60 -> 0.29), k margin,
    and a CALIBRATED-IV variant (pre_iv = trailing realized vol). NOTE the
    calibrated variant is a VRP-stripped lower bound, NOT corroboration of edge:
    it prices the option at realized vol with ZERO volatility risk premium, which
    removes the exact drag that makes long earnings vol lose, so its profitability
    is circular (the IV is anchored to the same realized vol the gate screens on).

Writes results/thorough_backtest.json (full per-event + per-ticker + pooled +
sensitivities) and prints a compact summary. Also regenerates the canonical
results/{metrics,validation}.json + equity_curve.png over the thorough universe
by monkeypatching the model module's WATCHLIST/START/END and calling main().
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglib.backtest.metrics import benjamini_hochberg_fdr, bootstrap_t_test, trade_metrics
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.events.earnings import get_earnings_dates
from tradinglib.options.surface import realized_vol

_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "options" / "03-earnings-straddle-spy"
)


def _load_backtest_module():
    spec = importlib.util.spec_from_file_location("es_backtest", _MODEL_DIR / "backtest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = _load_backtest_module()

UNIVERSE = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "NFLX", "AMD"]
START = "2020-01-01"
END = "2026-06-05"
CRUSH_RATIO = 0.25 / 0.45  # keep the post/pre crush proportion fixed across the pre_iv sweep


def _load(ticker: str):
    bars = load_daily(ticker, START, END)
    close = bars["close"]
    close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
    ev = get_earnings_dates([ticker], start=START, end=END)["earnings_datetime"]
    ev = pd.to_datetime(ev, utc=True).sort_values()
    ev_naive = ev.dt.tz_localize(None)
    usable = [
        e for e in ev_naive if close.index[bt.ENTRY_LEAD] <= e <= close.index[-bt.EXIT_OFFSET - 1]
    ]
    return close, ev, usable


def _event_rows(ticker, close, ev_aware, usable, *, k, lookback, pre_iv, post_iv, calibrated=False):
    """One row per usable event: gate inputs, fire flag, filtered & unfiltered P&L."""
    rows = []
    rv = realized_vol(close, window=21) if calibrated else None
    for traded in usable:
        prior = ev_aware[ev_aware < pd.Timestamp(traded, tz="UTC")]
        e_pre, e_post = pre_iv, post_iv
        if calibrated:
            v = float(rv.asof(traded)) if rv is not None else float("nan")
            if not np.isfinite(v) or v <= 0:
                continue
            e_pre = v
            e_post = v * CRUSH_RATIO
        try:
            rep = bt.run_synthetic(
                close=close,
                earnings_datetime=traded,
                pre_iv=e_pre,
                post_iv=e_post,
                k=k,
                lookback=lookback,
                prior_earnings=prior,
            )
        except ValueError:
            continue
        # realized overnight earnings move (same definition expected_move uses)
        bars = list(close.index)
        e_idx = next((i for i, b in enumerate(bars) if b >= traded), None)
        realized = float("nan")
        if e_idx is not None and e_idx >= 1:
            realized = abs(float(close.iloc[e_idx]) / float(close.iloc[e_idx - 1]) - 1.0)
        rows.append(
            {
                "ticker": ticker,
                "date": str(pd.Timestamp(traded).date()),
                "implied_move": rep["implied_move"],
                "expected_move": rep["expected_move"],
                "realized_move": realized,
                "gate_fired": bool(rep["filtered"]["took_trade"]),
                "filtered_pnl": rep["filtered"]["trade_pnl"]
                if rep["filtered"]["took_trade"]
                else None,
                "unfiltered_pnl": rep["unfiltered"]["trade_pnl"],
            }
        )
    return rows


def _pool(rows, which):
    key = "filtered_pnl" if which == "filtered" else "unfiltered_pnl"
    vals = [r[key] for r in rows if r[key] is not None]
    s = pd.Series(vals, dtype=float)
    tm = trade_metrics(s)
    t_stat, lo, hi, p = bootstrap_t_test(s, n_boot=2000, seed=0)
    return {
        "n": len(vals),
        "total_pnl": float(s.sum()) if len(vals) else 0.0,
        "trade_metrics": tm,
        "bootstrap": {"t_stat": t_stat, "ci_lo": lo, "ci_hi": hi, "p_value": p},
    }


def main():
    loaded = {t: _load(t) for t in UNIVERSE}

    # ---- baseline event study (model defaults) ----
    baseline_rows = []
    per_ticker = {}
    for t in UNIVERSE:
        close, ev_aware, usable = loaded[t]
        rows = _event_rows(t, close, ev_aware, usable, k=1.2, lookback=8, pre_iv=0.45, post_iv=0.25)
        baseline_rows.extend(rows)
        per_ticker[t] = {
            "n_events": len(rows),
            "n_gate_fired": sum(r["gate_fired"] for r in rows),
            "filtered": _pool(rows, "filtered"),
            "unfiltered": _pool(rows, "unfiltered"),
            "mean_realized_move": float(
                np.nanmean([r["realized_move"] for r in rows]) if rows else float("nan")
            ),
            "mean_implied_move": float(
                np.nanmean([r["implied_move"] for r in rows]) if rows else float("nan")
            ),
        }

    pooled_filtered = _pool(baseline_rows, "filtered")
    pooled_unfiltered = _pool(baseline_rows, "unfiltered")

    # FDR across tickers on the filtered branch
    fdr_tickers = [t for t in UNIVERSE if per_ticker[t]["filtered"]["n"] >= 2]
    pvals = [per_ticker[t]["filtered"]["bootstrap"]["p_value"] for t in fdr_tickers]
    rejected, thr = benjamini_hochberg_fdr(pvals, alpha=0.05)

    # ---- sensitivity: pre_iv level (proportional crush) ----
    pre_iv_sweep = []
    for pre in [0.30, 0.40, 0.45, 0.55, 0.65]:
        post = round(pre * CRUSH_RATIO, 4)
        rows = []
        for t in UNIVERSE:
            close, ev_aware, usable = loaded[t]
            rows.extend(
                _event_rows(t, close, ev_aware, usable, k=1.2, lookback=8, pre_iv=pre, post_iv=post)
            )
        f = _pool(rows, "filtered")
        pre_iv_sweep.append(
            {
                "pre_iv": pre,
                "post_iv": post,
                "n_gate_fired": f["n"],
                "n_events": len(rows),
                "expectancy": f["trade_metrics"]["expectancy"],
                "win_rate": f["trade_metrics"]["win_rate"],
                "profit_factor": f["trade_metrics"]["profit_factor"],
                "total_pnl": f["total_pnl"],
                "p_value": f["bootstrap"]["p_value"],
            }
        )

    # ---- sensitivity: k margin (reuse baseline pre/post; recompute gate per k) ----
    k_sweep = []
    for k in [1.05, 1.2, 1.5, 2.0]:
        rows = []
        for t in UNIVERSE:
            close, ev_aware, usable = loaded[t]
            rows.extend(
                _event_rows(t, close, ev_aware, usable, k=k, lookback=8, pre_iv=0.45, post_iv=0.25)
            )
        f = _pool(rows, "filtered")
        k_sweep.append(
            {
                "k": k,
                "n_gate_fired": f["n"],
                "expectancy": f["trade_metrics"]["expectancy"],
                "win_rate": f["trade_metrics"]["win_rate"],
                "profit_factor": f["trade_metrics"]["profit_factor"],
                "total_pnl": f["total_pnl"],
                "p_value": f["bootstrap"]["p_value"],
            }
        )

    # ---- sensitivity: calibrated IV (pre_iv = trailing realized vol * 1.0) ----
    calib_rows = []
    for t in UNIVERSE:
        close, ev_aware, usable = loaded[t]
        calib_rows.extend(
            _event_rows(
                t,
                close,
                ev_aware,
                usable,
                k=1.2,
                lookback=8,
                pre_iv=0.45,
                post_iv=0.25,
                calibrated=True,
            )
        )
    calib_filtered = _pool(calib_rows, "filtered")
    calib_unfiltered = _pool(calib_rows, "unfiltered")
    calib_fired = sum(r["gate_fired"] for r in calib_rows)

    out = {
        "universe": UNIVERSE,
        "window": [START, END],
        "n_events_total": len(baseline_rows),
        "baseline": {
            "params": {"k": 1.2, "lookback": 8, "pre_iv": 0.45, "post_iv": 0.25},
            "pooled_filtered": pooled_filtered,
            "pooled_unfiltered": pooled_unfiltered,
            "n_gate_fired": sum(r["gate_fired"] for r in baseline_rows),
            "fdr": {
                "tickers": fdr_tickers,
                "p_values": pvals,
                "rejected": rejected,
                "threshold": thr,
            },
            "per_ticker": per_ticker,
        },
        "sensitivity_pre_iv": pre_iv_sweep,
        "sensitivity_k": k_sweep,
        "sensitivity_calibrated_iv": {
            "n_events": len(calib_rows),
            "n_gate_fired": calib_fired,
            "pooled_filtered": calib_filtered,
            "pooled_unfiltered": calib_unfiltered,
        },
        "events": baseline_rows,
    }

    out_path = _MODEL_DIR / "results" / "thorough_backtest.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    # ---- compact console summary ----
    def line(label, p):
        tm = p["trade_metrics"]
        b = p["bootstrap"]
        print(
            f"{label:>26s}: n={p['n']:>3d}  exp=${tm['expectancy']:>9.2f}  "
            f"win={tm['win_rate']:.2%}  PF={tm['profit_factor']:.2f}  "
            f"total=${p['total_pnl']:>11.2f}  p={b['p_value']:.3f}  "
            f"CI=[{b['ci_lo']:.1f},{b['ci_hi']:.1f}]"
        )

    print(f"\n=== THOROUGH BACKTEST  universe={len(UNIVERSE)}  events={len(baseline_rows)} ===")
    print(f"gate fired (baseline k=1.2, pre_iv=0.45): {out['baseline']['n_gate_fired']}")
    line("POOLED filtered", pooled_filtered)
    line("POOLED unfiltered", pooled_unfiltered)
    print("\n--- per ticker (filtered) ---")
    for t in UNIVERSE:
        pt = per_ticker[t]
        print(
            f"{t:>6s}: events={pt['n_events']:>2d} fired={pt['n_gate_fired']:>2d}  "
            f"mean|realized|={pt['mean_realized_move']:.3f} mean_implied={pt['mean_implied_move']:.3f}"
        )
        line(f"  {t} filtered", pt["filtered"])
    print(
        f"\nFDR tickers={fdr_tickers}\n   p={[round(x, 3) for x in pvals]}\n   rejected={rejected} thr={thr}"
    )
    print("\n--- sensitivity: pre_iv (proportional crush) ---")
    for r in pre_iv_sweep:
        print(
            f"  pre_iv={r['pre_iv']:.2f} post={r['post_iv']:.3f}: fired={r['n_gate_fired']:>3d}/{r['n_events']}"
            f"  exp=${r['expectancy']:>8.2f} win={r['win_rate']:.2%} PF={r['profit_factor']:.2f}"
            f" total=${r['total_pnl']:>10.2f} p={r['p_value']:.3f}"
        )
    print("\n--- sensitivity: k margin (pre_iv=0.45) ---")
    for r in k_sweep:
        print(
            f"  k={r['k']:.2f}: fired={r['n_gate_fired']:>3d}  exp=${r['expectancy']:>8.2f}"
            f" win={r['win_rate']:.2%} PF={r['profit_factor']:.2f} total=${r['total_pnl']:>10.2f} p={r['p_value']:.3f}"
        )
    print("\n--- sensitivity: CALIBRATED IV (pre_iv = trailing realized vol) ---")
    print(f"  gate fired={calib_fired}/{len(calib_rows)}")
    line("  calib filtered", calib_filtered)
    line("  calib unfiltered", calib_unfiltered)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
