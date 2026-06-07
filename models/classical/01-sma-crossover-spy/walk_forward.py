"""Walk-forward validation of the SMA crossover, re-optimizing fast/slow.

Re-selects the (fast, slow) pair on each anchored in-sample window and reports
the stitched out-of-sample equity deflated by the parameter-grid size — the
honest counterpart to the single full-sample run in ``backtest.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from backtest import END, START, SYMBOL, build_signal  # noqa: E402

from tradinglib.backtest import run_backtest  # noqa: E402
from tradinglib.loaders.equities.yfinance import load_daily  # noqa: E402
from tradinglib.validation import walk_forward  # noqa: E402

RESULTS = HERE / "results"
GRID = {"fast": [10, 20, 50], "slow": [100, 150, 200]}


def make_signal(train: pd.DataFrame, test: pd.DataFrame, params: dict) -> pd.Series:
    # Warm the rolling means with train history straddling the boundary. Dedupe
    # so in-sample scoring (where test == train) doesn't double the index.
    closes = pd.concat([train["close"], test["close"]])
    closes = closes[~closes.index.duplicated(keep="last")]
    sig = build_signal(closes, fast=params["fast"], slow=params["slow"])
    return sig.loc[test.index]


def main() -> None:
    bars = load_daily(SYMBOL, start=START, end=END)
    data = bars[["open", "close"]].dropna()

    wf = walk_forward(
        data, make_signal, param_grid=GRID, mode="anchored",
        initial_train=756, test_size=126, step=126,  # ~3y train, ~6mo OOS
    )

    # Naive full-sample Sharpe (the optimistic number we are correcting).
    naive_sig = build_signal(data["close"], fast=50, slow=200)
    naive = run_backtest(data["close"], naive_sig, open_prices=data["open"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_windows": len(wf.windows),
        "n_trials": int(wf.oos_result.config["n_trials"]),
        "walk_forward_oos": wf.oos_result.metrics,
        "naive_full_sample_sharpe": naive.metrics["sharpe"],
        "param_stability": wf.param_stability,
    }
    (RESULTS / "walk_forward.json").write_text(json.dumps(summary, indent=2, default=str))
    wf.windows.to_csv(RESULTS / "walk_forward_windows.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    wf.oos_result.equity_curve.plot(ax=ax, label="SMA walk-forward (OOS)")
    ax.set_title(f"{SYMBOL} — SMA crossover, walk-forward OOS")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS / "walk_forward_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
