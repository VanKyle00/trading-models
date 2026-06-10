"""Tests for the Russell 1000 constituent-universe loader."""

from __future__ import annotations

from io import StringIO

import pandas as pd

_FAKE_WIKI_HTML = """
<html><body>
<table>
  <thead>
    <tr><th>Company</th><th>Symbol</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
  </thead>
  <tbody>
    <tr><td>Apple Inc.</td><td>AAPL</td><td>Information Technology</td>
        <td>Technology Hardware</td></tr>
    <tr><td>Berkshire Hathaway</td><td>BRK.B</td><td>Financials</td>
        <td>Multi-Sector Holdings</td></tr>
    <tr><td>Newco Without Cik</td><td>NEWCO</td><td>Industrials</td><td>Widgets</td></tr>
  </tbody>
</table>
</body></html>
"""

_CIK_MAP = {"AAPL": 320193, "BRK-B": 1067983}


class _FakeResponse:
    text = _FAKE_WIKI_HTML

    def raise_for_status(self) -> None:
        pass


def test_canonicalize_schema_cik_join_and_normalization() -> None:
    from tradinglib.loaders.universe import russell1000 as loader

    raw = pd.read_html(StringIO(_FAKE_WIKI_HTML))[0]
    out = loader._canonicalize(raw, _CIK_MAP)

    assert list(out.columns) == ["ticker", "name", "sector", "sub_industry", "cik"]
    assert out["cik"].dtype == "Int64"  # nullable: not every ticker maps
    indexed = out.set_index("ticker")
    assert indexed.loc["BRK-B", "cik"] == 1067983  # BRK.B -> BRK-B before the join
    assert pd.isna(indexed.loc["NEWCO", "cik"])  # unmapped ticker kept, cik <NA>
    assert indexed.loc["AAPL", "sector"] == "Information Technology"
