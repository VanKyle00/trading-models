"""Nightly regime overlay: benchmark trend + vol state, and the v1 gate.

v1 gates ONE failure mode, the one the 2025-08..2026-06 replay caught:
mean-reversion entries issued against the benchmark trend. Trend and
event styles are deliberately ungated until forward evidence argues
otherwise. ``neutral`` (insufficient history / data failure) never gates —
a missing benchmark must not silence the scanner.
"""

from __future__ import annotations

import pandas as pd

# Every tournament strategy key and every setup:* label the pipeline can issue.
STRATEGY_STYLES: dict[str, str] = {
    "sma_cross": "trend",
    "donchian": "trend",
    "macd": "trend",
    "ridge_momentum": "trend",
    "base_breakout": "trend",
    # Deliberately "trend" despite the registry's mean_reversion bucket: the rule
    # requires an existing uptrend and exits on trend failure; the replay failure
    # mode was rsi2/bollinger dip-buys only.
    "ma_pullback": "trend",
    "rsi2": "meanrev",
    "bollinger": "meanrev",
    "pead": "event",
    "setup:base_breakout": "trend",
    "setup:ma_pullback": "trend",
    "setup:pead": "event",
    "setup:base_breakdown": "trend",
    "setup:ma_rally_fade": "trend",
    "setup:pead_down": "event",
}


def regime_state(benchmark_close: pd.Series | None) -> dict:
    """The night's regime from benchmark closes; neutral when unknowable.

    NaN closes are dropped; the 220-bar history floor counts valid bars only.
    """
    if benchmark_close is not None:
        benchmark_close = benchmark_close.dropna()
    if benchmark_close is None or len(benchmark_close) < 220:
        return {"trend": "neutral", "close": None, "sma200": None, "vol_pctile": None}
    close = float(benchmark_close.iloc[-1])
    sma200 = float(benchmark_close.rolling(200).mean().iloc[-1])
    rets = benchmark_close.pct_change()
    vol20 = rets.rolling(20).std()
    vol_pctile = float((vol20.iloc[-252:].dropna() <= vol20.iloc[-1]).mean())
    return {
        "trend": "up" if close > sma200 else "down",
        "close": close,
        "sma200": sma200,
        "vol_pctile": vol_pctile,
    }


def gate_reason(strategy: str, stance: str, regime: dict) -> str | None:
    """Why this issuance is blocked in this regime, or None to allow."""
    if STRATEGY_STYLES.get(strategy) != "meanrev":
        return None
    trend = regime.get("trend")
    if stance == "long" and trend == "down":
        return "meanrev long in down-trend regime"
    if stance == "short" and trend == "up":
        return "meanrev short in up-trend regime"
    return None
