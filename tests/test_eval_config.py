from tradinglib.assistant_eval.config import DEFAULT_BAR, ShipBar


def test_default_bar_matches_spec():
    assert DEFAULT_BAR.tool_call_accuracy == 0.90
    assert DEFAULT_BAR.grounded_fraction == 0.99
    assert DEFAULT_BAR.judge_winrate == 0.45


def test_ship_bar_is_overridable():
    bar = ShipBar(tool_call_accuracy=0.8, grounded_fraction=1.0, judge_winrate=0.5)
    assert bar.grounded_fraction == 1.0
