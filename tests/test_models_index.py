"""Tests for the MODELS.md index generator and the underlying discovery
helpers in :mod:`tradinglib.models_index`."""

from __future__ import annotations

import sys
from pathlib import Path

from tradinglib.models_index import find_models, parse_frontmatter

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from regenerate_models_index import (  # noqa: E402
    fmt,
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


def test_find_models_scans_directory_tree(tmp_path: Path, monkeypatch) -> None:
    # Redirect repo_root so the path-relativization works against tmp_path
    from tradinglib import models_index as mi

    monkeypatch.setattr(mi, "repo_root", lambda: tmp_path)

    a = tmp_path / "models" / "classical" / "01-alpha"
    b = tmp_path / "models" / "ml" / "01-beta"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "model.md").write_text(
        "---\nname: Alpha\nfamily: classical\n---\nbody\n", encoding="utf-8"
    )
    (b / "model.md").write_text("---\nname: Beta\nfamily: ml\n---\nbody\n", encoding="utf-8")

    rows = find_models(tmp_path / "models")
    assert len(rows) == 2
    names = {r["name"] for r in rows}
    assert names == {"Alpha", "Beta"}
    # _path is set relative to repo_root() with forward slashes
    paths = {r["_path"] for r in rows}
    assert paths == {"models/classical/01-alpha", "models/ml/01-beta"}


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
