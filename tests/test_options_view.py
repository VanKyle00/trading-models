"""Tests for the options GUI view guards."""

from __future__ import annotations

import unittest.mock as mock

import plotly.graph_objects as go

import app.ui.options_view as ov
from app.ui import options_view


def test_render_is_noop_without_options_keys() -> None:
    # No payoff/simulation keys -> guards short-circuit, no Streamlit calls made.
    options_view.render({"result": object(), "data": object()})


def test_render_payoff_branch_invokes_chart(monkeypatch) -> None:
    mock_st = mock.MagicMock()
    mock_st.columns.return_value = (
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )
    monkeypatch.setattr(ov, "st", mock_st)
    monkeypatch.setattr(go, "Figure", mock.MagicMock(return_value=mock.MagicMock()))
    payoff = {"spots": [1, 2], "values": [3, 4], "expiry_values": [0, 1], "strike": 1.5}
    ov.render({"payoff": payoff})
    ov.st.plotly_chart.assert_called_once()


def test_render_simulation_branch_invokes_chart(monkeypatch) -> None:
    mock_st = mock.MagicMock()
    mock_st.columns.return_value = (
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    )
    monkeypatch.setattr(ov, "st", mock_st)
    monkeypatch.setattr(go, "Figure", mock.MagicMock(return_value=mock.MagicMock()))
    sim = mock.MagicMock()
    sim.truncated = False
    sim.prob_of_profit = 0.5
    sim.mean = 100.0
    sim.expected_shortfall = -50.0
    sim.percentiles = {5: -100.0, 25: -20.0, 50: 10.0, 75: 40.0, 95: 120.0}
    sim.pnl_distribution = [1.0, 2.0, 3.0]
    ov.render({"simulation": sim})
    ov.st.plotly_chart.assert_called_once()
