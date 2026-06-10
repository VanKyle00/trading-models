"""Tests for forward ticket evaluation (paper-trading the levels on daily bars)."""

from __future__ import annotations

import pandas as pd
import pytest

from tradinglib.strategist.evaluate import simulate_ticket

ASOF = "2026-06-10"  # bars below start strictly after this date


def _bars(rows: list[tuple[float, float, float, float]], tz: str | None = "UTC") -> pd.DataFrame:
    """Synthetic daily bars starting 2026-06-11 (a Thursday), business-day index."""
    idx = pd.date_range("2026-06-11", periods=len(rows), freq="B", tz=tz)
    o, h, lo, c = zip(*rows, strict=True)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


def _ticket(
    entry: float = 100.0,
    entry_type: str = "market",
    stop: float = 95.0,
    target: float = 110.0,
    stance: str = "long",
) -> dict:
    return {
        "ticker": "TST",
        "stance": stance,
        "strategy": "sma_cross",
        "levels": {
            "entry": entry,
            "entry_type": entry_type,
            "stop": stop,
            "target": target,
            "condition": "test",
        },
    }


def test_market_entry_fills_at_next_open() -> None:
    out = simulate_ticket(_ticket(), _bars([(101.0, 103.0, 100.0, 102.0)]), asof=ASOF)
    assert out["status"] == "open"
    assert out["entry_date"] == "2026-06-11"
    assert out["entry_fill"] == 101.0
    assert out["r"] == pytest.approx((102.0 - 101.0) / 5.0)  # unrealized, planned risk = 5
    assert out["pct_move"] == pytest.approx(1.0 / 101.0)
    assert out["sessions_held"] == 1
    assert out["closes"] == [["2026-06-11", 102.0]]


def test_stop_entry_waits_for_trigger_then_fills_at_level() -> None:
    bars = _bars([(98.0, 99.0, 97.0, 98.0), (99.0, 101.0, 98.0, 100.5)])
    out = simulate_ticket(_ticket(entry_type="stop"), bars, asof=ASOF)
    assert out["status"] == "open"
    assert out["entry_date"] == "2026-06-12"
    assert out["entry_fill"] == 100.0  # high crossed entry; level fill, not the high
    assert out["r"] == pytest.approx(0.5 / 5.0)


def test_stop_entry_gap_through_fills_at_open() -> None:
    out = simulate_ticket(
        _ticket(entry_type="stop"), _bars([(103.0, 104.0, 102.0, 103.0)]), asof=ASOF
    )
    assert out["entry_fill"] == 103.0  # gapped past the trigger: filled worse, at the open


def test_limit_entry_fills_on_dip() -> None:
    bars = _bars([(102.0, 103.0, 101.0, 102.0), (101.0, 102.0, 99.5, 101.0)])
    out = simulate_ticket(_ticket(entry_type="limit"), bars, asof=ASOF)
    assert out["entry_date"] == "2026-06-12"
    assert out["entry_fill"] == 100.0


def test_limit_entry_gap_through_fills_at_open() -> None:
    out = simulate_ticket(_ticket(entry_type="limit"), _bars([(98.0, 99.0, 97.0, 98.5)]), asof=ASOF)
    assert out["entry_fill"] == 98.0  # gapped below the limit: filled better, at the open


def test_waiting_inside_entry_window() -> None:
    bars = _bars([(98.0, 99.0, 97.0, 98.0)] * 3)
    out = simulate_ticket(_ticket(entry_type="stop"), bars, asof=ASOF)
    assert out["status"] == "waiting"
    assert out["entry_fill"] is None
    assert out["r"] is None
    assert len(out["closes"]) == 3


def test_expired_after_entry_window() -> None:
    bars = _bars([(98.0, 99.0, 97.0, 98.0)] * 7)
    out = simulate_ticket(_ticket(entry_type="stop"), bars, asof=ASOF)
    assert out["status"] == "expired"
    assert out["r"] is None
    assert len(out["closes"]) == 5  # sparkline capped at the entry window


def test_target_hit_fills_at_level() -> None:
    bars = _bars(
        [
            (100.0, 101.0, 99.0, 100.5),
            (105.0, 109.0, 104.0, 108.0),
            (109.0, 111.0, 108.0, 110.5),
        ]
    )
    out = simulate_ticket(_ticket(), bars, asof=ASOF)
    assert out["status"] == "target"
    assert out["exit_date"] == "2026-06-15"  # third business day
    assert out["exit_fill"] == 110.0
    assert out["r"] == pytest.approx(2.0)
    assert out["sessions_held"] == 3
    assert out["ambiguous_bar"] is False
    assert len(out["closes"]) == 3


def test_target_gap_through_fills_at_open() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0), (112.0, 113.0, 111.0, 112.5)])
    out = simulate_ticket(_ticket(), bars, asof=ASOF)
    assert out["status"] == "target"
    assert out["exit_fill"] == 112.0  # gapped past the target: filled better, at the open
    assert out["r"] == pytest.approx(2.4)


def test_stop_gap_through_fills_at_open_worse() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0), (93.0, 94.0, 92.0, 93.5)])
    out = simulate_ticket(_ticket(), bars, asof=ASOF)
    assert out["status"] == "stopped"
    assert out["exit_fill"] == 93.0  # gapped below the stop: filled worse, at the open
    assert out["r"] == pytest.approx(-1.4)


def test_ambiguous_bar_counts_as_stopped() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0), (100.0, 111.0, 94.0, 100.0)])
    out = simulate_ticket(_ticket(), bars, asof=ASOF)
    assert out["status"] == "stopped"  # both levels touched; worst case wins
    assert out["ambiguous_bar"] is True
    assert out["exit_fill"] == 95.0
    assert out["r"] == pytest.approx(-1.0)


def test_exit_can_trigger_on_the_entry_bar() -> None:
    out = simulate_ticket(_ticket(), _bars([(100.0, 101.0, 94.0, 95.0)]), asof=ASOF)
    assert out["status"] == "stopped"
    assert out["entry_date"] == out["exit_date"] == "2026-06-11"
    assert out["sessions_held"] == 1


def test_open_ticket_reports_unrealized_r() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0), (104.0, 106.0, 103.0, 105.0)])
    out = simulate_ticket(_ticket(), bars, asof=ASOF)
    assert out["status"] == "open"
    assert out["r"] == pytest.approx(1.0)
    assert out["sessions_held"] == 2


def test_short_stop_entry_and_protective_stop_mirror() -> None:
    # short: enter when price falls to 100; protective stop above at 105, target 90
    ticket = _ticket(entry=100.0, entry_type="stop", stop=105.0, target=90.0, stance="short")
    bars = _bars([(101.0, 102.0, 99.0, 100.0), (104.0, 106.0, 103.0, 105.0)])
    out = simulate_ticket(ticket, bars, asof=ASOF)
    assert out["entry_fill"] == 100.0
    assert out["status"] == "stopped"  # high crossed the protective stop
    assert out["exit_fill"] == 105.0
    assert out["r"] == pytest.approx(-1.0)
    assert out["pct_move"] == pytest.approx(-0.05)


def test_short_target_hit() -> None:
    ticket = _ticket(entry=100.0, entry_type="market", stop=105.0, target=90.0, stance="short")
    bars = _bars([(100.0, 101.0, 99.0, 100.0), (92.0, 93.0, 89.0, 90.0)])
    out = simulate_ticket(ticket, bars, asof=ASOF)
    assert out["status"] == "target"
    assert out["exit_fill"] == 90.0
    assert out["r"] == pytest.approx(2.0)


def test_no_sessions_after_asof_is_waiting() -> None:
    out = simulate_ticket(_ticket(), _bars([(100.0, 101.0, 99.0, 100.0)]), asof="2026-06-13")
    assert out["status"] == "waiting"
    assert out["closes"] == []


def test_naive_index_supported() -> None:
    out = simulate_ticket(_ticket(), _bars([(101.0, 103.0, 100.0, 102.0)], tz=None), asof=ASOF)
    assert out["status"] == "open"
    assert out["entry_fill"] == 101.0


def test_degenerate_levels_raise() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        simulate_ticket(
            _ticket(entry=100.0, stop=100.0), _bars([(100.0, 101.0, 99.0, 100.0)]), asof=ASOF
        )


def test_unknown_entry_type_raises() -> None:
    with pytest.raises(ValueError, match="entry_type"):
        simulate_ticket(
            _ticket(entry_type="teleport"), _bars([(100.0, 101.0, 99.0, 100.0)]), asof=ASOF
        )


def test_limit_entry_bar_target_only_touch_stays_open() -> None:
    # intrabar order unknowable: the rally to target may precede the dip that
    # filled the limit entry, so the entry bar's target touch must not exit
    bars = _bars([(102.0, 111.0, 99.5, 105.0)])
    out = simulate_ticket(_ticket(entry_type="limit"), bars, asof=ASOF)
    assert out["status"] == "open"
    assert out["entry_fill"] == 100.0
    assert out["r"] == pytest.approx(1.0)  # unrealized from the close


def test_limit_entry_bar_target_exit_allowed_next_bar() -> None:
    bars = _bars([(102.0, 111.0, 99.5, 105.0), (109.0, 111.0, 108.0, 110.0)])
    out = simulate_ticket(_ticket(entry_type="limit"), bars, asof=ASOF)
    assert out["status"] == "target"
    assert out["exit_date"] == "2026-06-12"
    assert out["exit_fill"] == 110.0


def test_limit_entry_bar_stop_hit_still_exits() -> None:
    # to reach the stop the price passed through the entry level first
    bars = _bars([(102.0, 103.0, 94.0, 95.0)])
    out = simulate_ticket(_ticket(entry_type="limit"), bars, asof=ASOF)
    assert out["status"] == "stopped"
    assert out["exit_fill"] == 95.0
    assert out["ambiguous_bar"] is False


def test_short_limit_entry_fills_on_rip() -> None:
    ticket = _ticket(entry=100.0, entry_type="limit", stop=105.0, target=90.0, stance="short")
    bars = _bars([(99.0, 101.0, 98.0, 99.0)])
    out = simulate_ticket(ticket, bars, asof=ASOF)
    assert out["status"] == "open"
    assert out["entry_fill"] == 100.0


def test_short_ambiguous_bar_counts_as_stopped() -> None:
    ticket = _ticket(entry=100.0, entry_type="market", stop=105.0, target=90.0, stance="short")
    bars = _bars([(100.0, 101.0, 99.0, 100.0), (100.0, 106.0, 89.0, 100.0)])
    out = simulate_ticket(ticket, bars, asof=ASOF)
    assert out["status"] == "stopped"
    assert out["ambiguous_bar"] is True
    assert out["exit_fill"] == 105.0
    assert out["r"] == pytest.approx(-1.0)


def test_nan_bars_are_dropped() -> None:
    bars = _bars([(101.0, 103.0, 100.0, 102.0), (float("nan"),) * 4, (104.0, 106.0, 103.0, 105.0)])
    out = simulate_ticket(_ticket(), bars, asof=ASOF)
    assert out["status"] == "open"
    assert out["sessions_held"] == 2  # NaN row ignored entirely
    assert out["r"] == pytest.approx((105.0 - 101.0) / 5.0)
