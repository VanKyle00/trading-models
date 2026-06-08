"""_report_view formats the earnings-straddle synthetic report for the results panel.

The earnings-straddle model stashes a filtered-vs-unfiltered comparison in
BacktestRun.extra['report']; this view turns it into display-ready rows. Other
models (no report) get None so the panel is simply skipped.
"""

from __future__ import annotations

from webapp.main import _report_view


def _sample_report() -> dict:
    return {
        "implied_move": 0.075,
        "expected_move": 0.031,
        "k": 1.2,
        "filtered": {"took_trade": False, "final_equity": 100_000.0, "trade_pnl": 0.0},
        "unfiltered": {"took_trade": True, "final_equity": 100_850.0, "trade_pnl": 850.0},
    }


def test_report_view_none_when_no_report() -> None:
    assert _report_view(None) is None


def test_report_view_formats_stats() -> None:
    view = _report_view(_sample_report())
    stats = {s["label"]: s["value"] for s in view["stats"]}
    assert stats["Implied move"] == "7.5%"
    assert stats["Expected move"] == "3.1%"
    assert stats["Edge margin k"] == "1.20"
    assert stats["Filter"] == "sat out"  # filtered.took_trade is False


def test_report_view_formats_branches() -> None:
    branches = _report_view(_sample_report())["branches"]
    assert branches[0]["name"].startswith("Filtered")
    assert branches[1]["name"].startswith("Unfiltered")
    assert branches[1]["final_equity"] == "$100,850"
    assert branches[1]["trade_pnl"] == "+$850"
    assert branches[1]["pnl_tone"] == "up"
    # zero P&L is not signed and reads as neutral
    assert branches[0]["trade_pnl"] == "$0"
    assert branches[0]["pnl_tone"] == "neutral"
