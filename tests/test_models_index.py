"""Tests for the MODELS.md index generator."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from regenerate_models_index import (  # noqa: E402
    fmt,
    parse_frontmatter,
    render,
    row_to_markdown,
)


def test_parse_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "model.md"
    p.write_text(
        "---\n"
        "name: Test\n"
        "family: classical\n"
        "window: swing\n"
        "assets: [equities]\n"
        "sharpe_oos: 1.5\n"
        "---\n"
        "\n"
        "Body text here.\n",
        encoding="utf-8",
    )
    meta = parse_frontmatter(p)
    assert meta is not None
    assert meta["name"] == "Test"
    assert meta["family"] == "classical"
    assert meta["assets"] == ["equities"]
    assert meta["sharpe_oos"] == 1.5


def test_parse_frontmatter_no_header(tmp_path: Path) -> None:
    p = tmp_path / "model.md"
    p.write_text("just plain text, no yaml\n", encoding="utf-8")
    assert parse_frontmatter(p) is None


def test_fmt_handles_types() -> None:
    assert fmt(None) == "—"
    assert fmt(["a", "b"]) == "a, b"
    assert fmt(1.234) == "1.23"
    assert fmt("status") == "status"


def test_row_to_markdown() -> None:
    meta = {
        "name": "SMA Crossover",
        "family": "classical",
        "window": "swing",
        "assets": ["equities"],
        "data_sources": ["yfinance"],
        "sharpe_oos": 0.73,
        "max_drawdown": -0.34,
        "status": "working",
        "_path": "models/classical/01-sma-crossover-spy",
    }
    row = row_to_markdown(meta)
    assert "[SMA Crossover](models/classical/01-sma-crossover-spy/)" in row
    assert "0.73" in row
    assert "-0.34" in row
    assert "working" in row


def test_render_empty() -> None:
    rendered = render([])
    assert "AUTO-GENERATED" in rendered
    assert "No models yet" in rendered


def test_render_with_rows() -> None:
    meta = {
        "name": "Test",
        "family": "classical",
        "_path": "models/classical/test",
    }
    rendered = render([meta])
    assert "No models yet" not in rendered
    assert "[Test]" in rendered
