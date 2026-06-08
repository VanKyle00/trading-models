"""The new model.md is discoverable and has the required frontmatter keys."""

from __future__ import annotations

from tradinglib.data.paths import repo_root
from tradinglib.models_index import find_models, parse_frontmatter


def test_model_md_frontmatter_has_required_keys() -> None:
    md = repo_root() / "models" / "options" / "03-earnings-straddle-spy" / "model.md"
    meta = parse_frontmatter(md)
    assert meta is not None
    for key in ("name", "family", "status", "sharpe_oos", "max_drawdown", "params"):
        assert key in meta
    assert meta["family"] == "options"


def test_find_models_discovers_earnings_straddle() -> None:
    paths = {m["_path"] for m in find_models()}
    assert "models/options/03-earnings-straddle-spy" in paths
