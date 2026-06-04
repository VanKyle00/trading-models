"""Tests for tradinglib.options pricing primitives."""

from __future__ import annotations

import math

import pytest


def test_package_imports() -> None:
    import tradinglib.options  # noqa: F401
