"""Risk-based sizing (spec C4): stock by stop distance, options by max loss,
CSP capital-sized, floor-to-zero reported unsized."""

from __future__ import annotations

from tradinglib.strategist.sizing import size_structure
from tradinglib.strategist.structures import Structure


def _structure(kind: str, unit: str, max_loss: float | None, legs: list | None = None) -> Structure:
    return Structure(
        kind=kind,
        label=kind,
        unit=unit,
        legs=legs or [],
        premium=None,
        max_loss=max_loss,
        max_gain=None,
        breakeven=None,
        pop_market_implied=None,
        rr=None,
        premium_yield=None,
    )


def test_stock_sized_by_stop_distance() -> None:
    s = _structure("stock", "share", max_loss=4.0)
    size_structure(s, account_size=100_000.0, risk_per_trade_pct=0.01)
    assert s.quantity == 250  # 1000 budget / $4 per share


def test_defined_risk_options_sized_by_max_loss_per_contract() -> None:
    s = _structure("bull_put_spread", "contract", max_loss=3.2)
    size_structure(s, account_size=100_000.0, risk_per_trade_pct=0.01)
    assert s.quantity == 3  # 1000 // 320


def test_floor_to_zero_is_reported_unsized_never_bumped() -> None:
    s = _structure("call_debit_spread", "contract", max_loss=12.0)
    size_structure(s, account_size=100_000.0, risk_per_trade_pct=0.01)
    assert s.quantity == 0
    assert any("unsized" in w for w in s.warnings)


def test_missing_max_loss_is_unsized_not_a_crash() -> None:
    s = _structure("long_call", "contract", max_loss=None)
    size_structure(s, account_size=100_000.0, risk_per_trade_pct=0.01)
    assert s.quantity == 0
    assert any("unsized" in w for w in s.warnings)


def test_csp_is_capital_sized() -> None:
    s = _structure("csp", "contract", max_loss=92.2, legs=[{"strike": 95.0}])
    size_structure(s, account_size=100_000.0, risk_per_trade_pct=0.01)
    assert s.quantity == 1  # floor(0.10 * 100k / (95 * 100))

    small = _structure("csp", "contract", max_loss=92.2, legs=[{"strike": 95.0}])
    size_structure(small, account_size=50_000.0, risk_per_trade_pct=0.01)
    assert small.quantity == 0
    assert any("unsized" in w for w in small.warnings)
