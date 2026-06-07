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
from datetime import date
from pathlib import Path
from typing import Any

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
FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
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


def run_for_gui(
    start: str | date = START,
    end: str | date = END,
    *,
    symbol: str = SYMBOL,
    window: int = ZSCORE_WINDOW,
    fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    initial_capital: float = 100_000.0,
    size_mult: float = 1.0,
) -> dict[str, Any]:
    """Trends contrarian backtest on a user-selected sub-window.

    Google Trends values are relative within the original query window
    (``TIMEFRAME``), so the loader always pulls the full 2021-2024 window
    and we slice the user's selection afterward. Z-score uses the full
    series for proper trailing context, then is sliced for evaluation.
    """
    if symbol != SYMBOL:
        raise ValueError(
            f"this model is locked to {SYMBOL} (paired with Google-Trends "
            f"'bitcoin' interest); got {symbol!r}"
        )
    daily = load_daily(SYMBOL, start=START, end=END)
    btc_weekly = daily["close"].resample("W-SUN").last().dropna()
    btc_weekly_open = daily["open"].resample("W-SUN").first().dropna()
    interest = load_interest(QUERY, timeframe=TIMEFRAME)

    common = btc_weekly.index.intersection(interest.index)
    btc_weekly = btc_weekly.loc[common]
    interest = interest.loc[common]

    signal_full = build_signal(interest, window=window)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mask = (btc_weekly.index >= start_ts) & (btc_weekly.index <= end_ts)
    btc_weekly = btc_weekly.loc[mask]
    interest = interest.loc[mask]
    signal = signal_full.loc[mask] * size_mult

    if len(btc_weekly) < 2:
        raise ValueError(
            f"only {len(btc_weekly)} weekly bars in [{start}, {end}] — pick a wider window"
        )

    # Every weekly close has a same-week open from the same daily frame, so the
    # reindex should never introduce a NaN. Assert it: a NaN open would be
    # silently zeroed by the engine's fill, masking a data gap.
    weekly_open = btc_weekly_open.reindex(btc_weekly.index)
    if weekly_open.isna().any():
        raise ValueError("weekly open series has gaps vs the weekly close index")
    result = run_backtest(
        btc_weekly,
        signal,
        open_prices=weekly_open,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        periods_per_year=52,
    )

    data = pd.DataFrame({"close": btc_weekly, "interest": interest, "signal": signal})
    return {
        "data": data,
        "result": result,
        "symbol": SYMBOL,
        "params": {
            "start": str(start),
            "end": str(end),
            "zscore_window": window,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "size_mult": size_mult,
        },
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = run_for_gui(START, END)
    result = out["result"]
    data = out["data"]

    metrics_path = RESULTS / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(f"Wrote {metrics_path}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    result.equity_curve.plot(ax=axes[0], label="Trends contrarian (BTC)")
    buy_hold = (1.0 + data["close"].pct_change().fillna(0.0)).cumprod() * result.config[
        "initial_capital"
    ]
    buy_hold.plot(ax=axes[0], label="Buy & hold (BTC weekly)", alpha=0.6)
    axes[0].set_title("BTC — Google-Trends contrarian vs buy & hold (weekly)")
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    data["interest"].plot(
        ax=axes[1], color="steelblue", alpha=0.7, label="'bitcoin' search interest"
    )
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
