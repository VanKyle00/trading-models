"""Tests for the filesystem layout helpers."""

from __future__ import annotations

from tradinglib.data.paths import processed_dir, raw_dir, repo_root


def test_repo_root_contains_pyproject() -> None:
    assert (repo_root() / "pyproject.toml").exists()


def test_processed_dir_path() -> None:
    assert processed_dir("yfinance") == repo_root() / "data" / "processed" / "yfinance"


def test_raw_dir_path() -> None:
    assert raw_dir("yfinance") == repo_root() / "data" / "raw" / "yfinance"
