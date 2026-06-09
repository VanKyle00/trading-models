"""Real-chain (Phase 2) backtest of model 03 — earnings event-vol straddle.

Quote-to-quote on DoltHub historical chains: entry buys the ATM straddle at
the real ask, exit sells at the real bid; the k-gate consumes the REAL implied
move (call_mid + put_mid)/spot, so the selection thesis is testable for the
first time (Phase 1's synthetic implied move was ~0.075 for every name).

Universe/window mirror the Phase-1 thorough backtest for comparability.
Skip reasons are counted and reported — no silent truncation. First live run
fetches ~2 chains per event from the DoltHub API (cached to parquet forever
after; politeness sleep 0.5 s per live call, so expect ~5-10 minutes cold).

Writes models/options/03-earnings-straddle-spy/results/real_chain_backtest.json
and prints a compact summary.

    uv run python scripts/earnings_straddle_real_chain_backtest.py
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
from tradinglib.loaders.options.dolthub import load_chain

_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "options" / "03-earnings-straddle-spy"
)

UNIVERSE = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "NFLX", "AMD"]
START = "2020-01-01"
END = "2026-06-05"
ENTRY_LEAD = 3
EXIT_OFFSET = 1
K = 1.2
LOOKBACK = 8
MAX_SPREAD_FRAC = 0.20
FEE_BPS = 1.0
K_GRID = [1.05, 1.2, 1.5, 2.0]
LOOKBACK_GRID = [4, 8, 12]


def _load_real_chain_module():
    spec = importlib.util.spec_from_file_location("es_real_chain", _MODEL_DIR / "real_chain.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_real_chain_module()


def _pool(pnls: list[float]) -> dict:
    s = pd.Series(pnls, dtype=float)
    tm = trade_metrics(s)
    t_stat, lo, hi, p = bootstrap_t_test(s, n_boot=2000, seed=0)
    return {
        "n": len(pnls),
        "total_pnl": float(s.sum()) if len(pnls) else 0.0,
        "trade_metrics": tm,
        "bootstrap": {"t_stat": t_stat, "ci_lo": lo, "ci_hi": hi, "p_value": p},
    }


def main() -> None:
    events: list[dict] = []
    skips: dict[str, int] = dict.fromkeys(rc.SKIP_REASONS, 0)
    n_window_skips = 0

    for ticker in UNIVERSE:
        bars = load_daily(ticker, START, END)
        close = bars["close"]
        close = pd.Series(close.to_numpy(), index=close.index.tz_localize(None), name="close")
        ev = get_earnings_dates([ticker], start=START, end=END)["earnings_datetime"]
        ev = pd.to_datetime(ev, utc=True).sort_values()
        ev = ev.drop_duplicates()

        for traded_aware in ev:
            traded = traded_aware.tz_localize(None)
            prior = ev[ev < traded_aware]
            prior_moves = rc.past_abs_moves(close, prior)
            try:
                rec = rc.run_event(
                    ticker=ticker,
                    close=close,
                    earnings_datetime=traded,
                    prior_moves=prior_moves,
                    load_chain=load_chain,
                    entry_lead=ENTRY_LEAD,
                    exit_offset=EXIT_OFFSET,
                    k=K,
                    lookback=LOOKBACK,
                    max_spread_frac=MAX_SPREAD_FRAC,
                    fee_bps=FEE_BPS,
                )
            except ValueError as err:
                if "window" not in str(err):
                    raise
                n_window_skips += 1
                continue
            except RuntimeError as err:
                # DoltHub API failure: annotate where the cold run died and
                # fail fast — the JSON must never cover an incomplete universe.
                raise RuntimeError(f"{ticker} {traded.date()}: {err}") from err
            if "skip_reason" in rec:
                skips[rec["skip_reason"]] += 1
                continue
            events.append(rec)
        print(f"{ticker}: {sum(1 for e in events if e['ticker'] == ticker)} events traded")

    filtered = [e["pnl"] for e in events if e["gate_fired"]]
    unfiltered = [e["pnl"] for e in events]

    per_ticker: dict[str, dict] = {}
    for ticker in UNIVERSE:
        t_events = [e for e in events if e["ticker"] == ticker]
        per_ticker[ticker] = {
            "n_events": len(t_events),
            "n_gate_fired": sum(e["gate_fired"] for e in t_events),
            "mean_implied_move": float(np.nanmean([e["implied_move"] for e in t_events]))
            if t_events
            else float("nan"),
            "filtered": _pool([e["pnl"] for e in t_events if e["gate_fired"]]),
            "unfiltered": _pool([e["pnl"] for e in t_events]),
        }

    fdr_tickers = [t for t in UNIVERSE if per_ticker[t]["filtered"]["n"] >= 2]
    pvals = [per_ticker[t]["filtered"]["bootstrap"]["p_value"] for t in fdr_tickers]
    rejected, threshold = benjamini_hochberg_fdr(pvals, alpha=0.05)

    sweep = []
    for k in K_GRID:
        for lb in LOOKBACK_GRID:
            pnls = rc.gate_pnls(events, k=k, lookback=lb)
            pool = _pool(pnls)
            sweep.append(
                {
                    "k": k,
                    "lookback": lb,
                    "n_fired": pool["n"],
                    "expectancy": pool["trade_metrics"]["expectancy"],
                    "win_rate": pool["trade_metrics"]["win_rate"],
                    "profit_factor": pool["trade_metrics"]["profit_factor"],
                    "total_pnl": pool["total_pnl"],
                    "p_value": pool["bootstrap"]["p_value"],
                }
            )

    out = {
        "phase": "2-real-chain-dolthub",
        "source": "dolthub post-no-preference/options (EOD quotes)",
        "universe": UNIVERSE,
        "window": [START, END],
        "params": {
            "k": K,
            "lookback": LOOKBACK,
            "entry_lead": ENTRY_LEAD,
            "exit_offset": EXIT_OFFSET,
            "max_spread_frac": MAX_SPREAD_FRAC,
            "fee_bps": FEE_BPS,
        },
        "n_events_traded": len(events),
        "n_gate_fired": len(filtered),
        "skips": {**skips, "window_out_of_range": n_window_skips},
        "pooled_filtered": _pool(filtered),
        "pooled_unfiltered": _pool(unfiltered),
        "fdr": {
            "tickers": fdr_tickers,
            "p_values": pvals,
            "rejected": rejected,
            "threshold": threshold,
        },
        "per_ticker": per_ticker,
        "sensitivity_k_lookback": sweep,
        "events": events,
    }

    out_path = _MODEL_DIR / "results" / "real_chain_backtest.json"
    out_path.write_text(
        json.dumps(rc.sanitize_for_json(out), indent=2, default=str, allow_nan=False)
    )

    def line(label: str, pool: dict) -> None:
        tm, b = pool["trade_metrics"], pool["bootstrap"]
        print(
            f"{label:>22s}: n={pool['n']:>3d}  exp=${tm['expectancy']:>9.2f}  "
            f"win={tm['win_rate']:.2%}  PF={tm['profit_factor']:.2f}  "
            f"total=${pool['total_pnl']:>11.2f}  p={b['p_value']:.3f}"
        )

    print(f"\n=== REAL-CHAIN BACKTEST  events={len(events)}  gate_fired={len(filtered)} ===")
    print(f"skips: {out['skips']}")
    line("POOLED filtered", out["pooled_filtered"])
    line("POOLED unfiltered", out["pooled_unfiltered"])
    print("\n--- per ticker (mean REAL implied move — must differ across names) ---")
    for t in UNIVERSE:
        pt = per_ticker[t]
        print(
            f"{t:>6s}: events={pt['n_events']:>2d} fired={pt['n_gate_fired']:>2d} "
            f"mean_implied={pt['mean_implied_move']:.4f}"
        )
    print(f"\nFDR: tickers={fdr_tickers} rejected={rejected} threshold={threshold}")
    print("\n--- sensitivity: k x lookback ---")
    for r in sweep:
        print(
            f"  k={r['k']:.2f} lb={r['lookback']:>2d}: fired={r['n_fired']:>3d} "
            f"exp=${r['expectancy']:>8.2f} win={r['win_rate']:.2%} "
            f"PF={r['profit_factor']:.2f} total=${r['total_pnl']:>10.2f} p={r['p_value']:.3f}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
