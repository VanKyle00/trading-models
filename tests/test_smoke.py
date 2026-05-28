"""Sanity check — verifies the package imports cleanly."""

import tradinglib


def test_version_string():
    assert tradinglib.__version__ == "0.1.0"
