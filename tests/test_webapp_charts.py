# tests/test_webapp_charts.py
import plotly.graph_objects as go

from tests._webapp_helpers import synthetic_run
from webapp.charts import BUILDERS, build, build_all


def test_equity_builder_returns_figure_with_two_traces():
    run = synthetic_run()
    fig = build("equity", run)
    assert isinstance(fig, go.Figure)
    names = {tr.name for tr in fig.data}
    assert {"Strategy", "Buy & hold"} <= names


def test_drawdown_builder_returns_figure():
    fig = build("drawdown", synthetic_run())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_build_all_returns_every_registered_builder():
    figs = build_all(synthetic_run())
    assert set(figs) == set(BUILDERS)
    assert all(isinstance(f, go.Figure) for f in figs.values())


def test_unknown_builder_raises():
    import pytest

    with pytest.raises(KeyError):
        build("does-not-exist", synthetic_run())


def test_all_expected_builders_registered():
    expected = {
        "equity",
        "drawdown",
        "rolling_sharpe",
        "returns_hist",
        "monthly_heatmap",
        "exposure",
        "trades_table",
    }
    assert expected <= set(BUILDERS)


def test_each_builder_produces_a_figure():
    run = synthetic_run()
    for name in BUILDERS:
        assert isinstance(build(name, run), go.Figure), name
