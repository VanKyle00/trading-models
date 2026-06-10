"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def make_chain():
    """Builder for synthetic option chains in the canonical schema + OI/volume.

    ``mids`` maps ``(right, dte, strike) -> mid``; bid/ask straddle the mid
    well inside the 10% liquidity gate unless overridden per row via
    ``overrides[(right, dte, strike)] = {"bid": ..., ...}``.
    """

    def _build(
        *,
        mids: dict[tuple[str, int, float], float],
        spot: float = 100.0,
        asof: str = "2026-06-10",
        iv: float = 0.30,
        spread_frac: float = 0.04,
        open_interest: float = 500.0,
        overrides: dict[tuple[str, int, float], dict] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for (right, dte, strike), mid in mids.items():
            row = {
                "date": pd.Timestamp(asof),
                "ticker": "TEST",
                "expiration": pd.Timestamp(asof) + pd.Timedelta(days=dte),
                "strike": float(strike),
                "right": right,
                "bid": mid * (1 - spread_frac / 2),
                "ask": mid * (1 + spread_frac / 2),
                "iv": iv,
                "spot": spot,
                "open_interest": open_interest,
                "volume": 100.0,
            }
            row.update((overrides or {}).get((right, dte, strike), {}))
            rows.append(row)
        return pd.DataFrame(rows)

    return _build
