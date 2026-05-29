"""Helpers for reading per-model GUI configuration from model.md frontmatter.

Each model declares which tickers it supports via the ``tickers`` key:
  - ``tickers: any``        → free-text yfinance symbol (e.g. classical)
  - ``tickers: [SPY]``      → locked to a single symbol
  - ``tickers: [SPY, QQQ]`` → a fixed choice list

These helpers translate that frontmatter into the three pieces the
sidebar needs: the input mode, the available choices, and a default.
"""

from __future__ import annotations

from typing import Any, Literal

TickerMode = Literal["free", "fixed", "choice"]


def ticker_mode(model: dict[str, Any]) -> TickerMode:
    """Return how the sidebar should render the ticker control."""
    tickers = model.get("tickers")
    if tickers == "any":
        return "free"
    if isinstance(tickers, (list, tuple)):
        return "fixed" if len(tickers) <= 1 else "choice"
    # Unspecified → treat as locked to whatever default we can find.
    return "fixed"


def ticker_choices(model: dict[str, Any]) -> list[str]:
    """Return the explicit ticker list, or ``[]`` for free-text models."""
    tickers = model.get("tickers")
    if isinstance(tickers, (list, tuple)):
        return [str(t) for t in tickers]
    return []


def default_ticker(model: dict[str, Any]) -> str:
    """Return the symbol the control should start on."""
    explicit = model.get("default_ticker")
    if explicit:
        return str(explicit)
    choices = ticker_choices(model)
    if choices:
        return choices[0]
    raise ValueError(
        f"model {model.get('name', '?')!r} has no default_ticker and no tickers list"
    )
