"""Tests for the options GUI view guards."""

from __future__ import annotations

from app.ui import options_view


def test_render_is_noop_without_options_keys() -> None:
    # No payoff/simulation keys -> guards short-circuit, no Streamlit calls made.
    options_view.render({"result": object(), "data": object()})
