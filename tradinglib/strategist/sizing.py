"""Risk-based position sizing (spec C4).

``risk_budget = account_size x risk_per_trade_pct``. Stock plans risk the
entry-to-stop distance per share; defined-risk option structures risk
``max_loss x 100`` per contract (a long option's max loss is its debit). The
CSP alone is capital-sized — 10% of the account in strike notional — because
its nominal stock-to-zero max loss would floor every account to zero
contracts. A quantity flooring to zero is reported ``unsized`` and never
silently bumped to 1.
"""

from __future__ import annotations

from tradinglib.strategist.structures import Structure

CSP_CAPITAL_FRAC = 0.10
SHARES_PER_CONTRACT = 100


def size_structure(s: Structure, *, account_size: float, risk_per_trade_pct: float) -> None:
    """Fill ``s.quantity`` in place; append the ``unsized`` warning on a zero floor."""
    budget = account_size * risk_per_trade_pct
    if s.kind == "csp":
        strike = float(s.legs[0]["strike"])
        qty = int(CSP_CAPITAL_FRAC * account_size // (strike * SHARES_PER_CONTRACT))
    elif s.max_loss is None or s.max_loss <= 0:
        qty = 0
    elif s.unit == "share":
        qty = int(budget // s.max_loss)
    else:
        qty = int(budget // (s.max_loss * SHARES_PER_CONTRACT))
    s.quantity = qty
    if qty == 0:
        s.warnings.append("unsized: one unit already exceeds the risk budget / capital slice")
