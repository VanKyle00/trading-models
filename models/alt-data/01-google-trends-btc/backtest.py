"""BTC weekly returns vs Google Trends 'bitcoin' search interest.

**Hypothesis.** Search interest in 'bitcoin' is mostly *coincident* with
price action — when BTC pumps, people search for it. Treated as a level,
search interest carries no useful information. But treated as an
*anomaly* against its recent average, extreme values may flag retail
attention spikes (FOMO → near-term mean reversion) and attention troughs
(disinterest → near-term recovery). This is a contrarian attention model.

**Signal.** 4-week rolling z-score of search interest:
- ``z < -1`` (low attention, contrarian) → long
- ``z > +1`` (overheated attention) → short
- Otherwise flat

**Data.** Weekly BTC-USD close prices from yfinance, resampled to
Sunday-anchored weeks to align with Google Trends' weekly index.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tradinglib.backtest import run_backtest
from tradinglib.loaders.equities.yfinance import load_daily
from tradinglib.loaders.sentiment.google_trends import load_interest

QUERY = "bitcoin"
SYMBOL = "BTC-USD"
TIMEFRAME = "2021-01-01 2024-12-31"
START = "2021-01-01"
END = "2024-12-31"
ZSCORE_WINDOW = 4
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def build_signal(interest: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Contrarian rule: short when attention is unusually high; long when low."""
    mean = interest.rolling(window).mean()
    std = interest.rolling(window).std()
    z = (interest - mean) / std

    signal = pd.Series(0.0, index=interest.index)
    signal[z < -1.0] = 1.0
    signal[z > 1.0] = -1.0
    return signal


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Weekly BTC close — yfinance gives daily; resample to Sunday-end weeks
    daily = load_daily(SYMBOL, start=START, end=END)
    btc_weekly = daily["close"].resample("W-SUN").last().dropna()

    interest = load_interest(QUERY, timeframe=TIMEFRAME)

    # Align on the common weekly index
    common = btc_weekly.index.intersection(interest.index)
    btc_weekly = btc_weekly.loc[common]
    interest = interest.loc[common]

    signal = build_signal(interest)

    # Weekly bars → 52 periods per year for annualization
    result = run_backtest(
        btc_weekly,
        signal,
        fee_bps=2.0,
        slippage_bps=3.0,
        periods_per_year=52,
    )

    metrics_path = RESULTS / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(f"Wrote {metrics_path}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    result.equity_curve.plot(ax=axes[0], label="Trends contrarian (BTC)")
    buy_hold = (1.0 + btc_weekly.pct_change().fillna(0.0)).cumprod() * result.config[
        "initial_capital"
    ]
    buy_hold.plot(ax=axes[0], label="Buy & hold (BTC weekly)", alpha=0.6)
    axes[0].set_title("BTC — Google-Trends contrarian vs buy & hold (weekly)")
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    interest.plot(ax=axes[1], color="steelblue", alpha=0.7, label="'bitcoin' search interest")
    axes[1].set_ylabel("Search interest")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plot_path = RESULTS / "equity_curve.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
