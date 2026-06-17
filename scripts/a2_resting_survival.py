"""A2 before/after survival experiment — resting-order tournament scoring.

Runs the per-ticker tournament twice on the SAME frozen bars — once on the
position-series path (``use_resting_engine=False``, today's behavior) and once on
the intrabar resting engine (``use_resting_engine=True``) — and diffs the
certified survivor set. Two things must hold for the change to be safe to ship:

1. ZERO BITS for the position-only strategies. Every strategy WITHOUT a
   ``resting_orders`` builder (sma_cross/rsi2/macd/ma_pullback/pead/ridge_momentum)
   must produce a byte-identical verdict on both runs. A nonzero diff there is a
   leak from the resting path into untouched strategies and FAILS the gate.

2. Every flip in the three resting strategies (donchian/base_breakout/bollinger)
   is reported with its before/after so a human can confirm each is explainable
   by "the engine now scores the resting stop/limit the ticket issues, not the
   close-cross/next-open proxy" — not a bug.

Data: real daily bars via the equities loader when reachable; otherwise a
deterministic seeded synthetic panel (printed in the header so a run is never
silently using the wrong data). Pure read/score — issues no tickets, writes no
ledger. Exit code is nonzero iff the zero-bits invariant is violated.

Usage:
    python scripts/a2_resting_survival.py                 # default ticker panel
    python scripts/a2_resting_survival.py --tickers AAPL MSFT NVDA
    python scripts/a2_resting_survival.py --synthetic     # force the offline panel
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tradinglib.tournament.run import TournamentConfig, TournamentResult, run_tournament
from tradinglib.tournament.strategies import STRATEGIES

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH",
    "JNJ", "V", "PG", "HD", "MA", "COST", "WMT", "BAC", "KO", "CVX",
    "AMD", "NFLX", "CRM", "INTC", "DIS",
]  # fmt: skip

RESTING_KEYS = tuple(k for k, s in STRATEGIES.items() if s.resting_orders is not None)
POSITION_KEYS = tuple(k for k, s in STRATEGIES.items() if s.resting_orders is None)
STANCES = ("long", "short")


def _synthetic_panel(tickers: list[str], n: int = 800) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLCV per ticker — varied drift/vol regimes so the
    resting strategies actually arm and trade (otherwise the diff is vacuous)."""
    idx = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    panel: dict[str, pd.DataFrame] = {}
    for seed, sym in enumerate(tickers):
        rng = np.random.default_rng(seed)
        drift = rng.uniform(-0.0004, 0.0008)
        vol = rng.uniform(0.012, 0.028)
        rets = rng.normal(drift, vol, n)
        close = 50.0 * np.cumprod(1.0 + rets)
        high = close * (1.0 + rng.uniform(0.0, 0.02, n))
        low = close * (1.0 - rng.uniform(0.0, 0.02, n))
        open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
        panel[sym] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
            },
            index=idx,
        )
    return panel


def _load_real(tickers: list[str], refresh: bool) -> dict[str, pd.DataFrame]:
    from tradinglib.loaders.equities.yfinance import load_daily

    panel: dict[str, pd.DataFrame] = {}
    for sym in tickers:
        try:
            bars = load_daily(sym, refresh=refresh)
        except Exception as exc:  # any loader/network failure -> skip this ticker
            print(f"  ! {sym}: load failed ({type(exc).__name__}: {exc})")
            continue
        if len(bars) >= TournamentConfig().initial_train + TournamentConfig().test_size:
            panel[sym] = bars
    return panel


def _verdicts(result: TournamentResult) -> dict[str, dict]:
    return {v.key: v.as_dict() for v in result.verdicts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--synthetic", action="store_true", help="force the offline panel")
    parser.add_argument("--refresh", action="store_true", help="redownload real bars")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    if args.synthetic:
        panel, source = _synthetic_panel(tickers), "SYNTHETIC (forced)"
    else:
        panel = _load_real(tickers, args.refresh)
        if panel:
            source = f"REAL daily bars ({len(panel)}/{len(tickers)} tickers loaded)"
        else:
            panel, source = _synthetic_panel(tickers), "SYNTHETIC (real load unavailable)"

    print("=" * 78)
    print("A2 resting-order survival experiment — before(position) vs after(resting)")
    print(f"data source : {source}")
    print(f"tickers     : {len(panel)}  |  stances: {', '.join(STANCES)}")
    print(f"resting     : {', '.join(RESTING_KEYS)}")
    print(f"position    : {', '.join(POSITION_KEYS)}")
    print("=" * 78)

    cfg_off = TournamentConfig()
    cfg_on = TournamentConfig(use_resting_engine=True)

    zero_bit_violations: list[str] = []
    resting_flips: list[str] = []
    position_compared = 0
    # per resting strategy: [#compared, #verdict changed by the engine]
    resting_moved = {k: [0, 0] for k in RESTING_KEYS}

    for sym, bars in panel.items():
        for stance in STANCES:
            off = _verdicts(run_tournament(bars, stance, config=cfg_off))
            on = _verdicts(run_tournament(bars, stance, config=cfg_on))

            for key in POSITION_KEYS:
                position_compared += 1
                if off.get(key) != on.get(key):
                    zero_bit_violations.append(
                        f"{sym}/{stance}/{key}: {off.get(key)} -> {on.get(key)}"
                    )

            for key in RESTING_KEYS:
                a, b = off.get(key), on.get(key)
                if a is None or b is None:
                    continue
                resting_moved[key][0] += 1
                if a != b:
                    resting_moved[key][1] += 1
                if a["survived"] != b["survived"]:
                    flag = "SURVIVED" if b["survived"] else "DROPPED"
                    resting_flips.append(
                        f"{sym}/{stance}/{key}: {flag}  "
                        f"dsr {a['deflated_sharpe']:.3f}->{b['deflated_sharpe']:.3f}  "
                        f"n_trades {a['n_trades']}->{b['n_trades']}"
                    )

    print(f"\nposition-only verdicts compared : {position_compared}")
    print(f"zero-bit violations              : {len(zero_bit_violations)}")
    for v in zero_bit_violations:
        print(f"  X {v}")

    print("\nresting-strategy effect (verdict changed by the engine / compared):")
    for key, (compared, moved) in resting_moved.items():
        print(f"  {key:<14} {moved}/{compared} re-scored")

    print(f"\nresting-strategy survival flips  : {len(resting_flips)}")
    for f in resting_flips:
        print(f"  ~ {f}")
    if not resting_flips:
        print("  (no survival flips on this panel — expected when the whole DSR")
        print("   distribution sits far below the 0.90 gate; the re-scored counts")
        print("   above confirm the engine still changed the underlying metrics)")

    ok = not zero_bit_violations
    print(
        "\n"
        + ("PASS" if ok else "FAIL")
        + ": position-only strategies are "
        + (
            "byte-identical across the flag."
            if ok
            else "NOT byte-identical — leak into untouched strategies."
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
